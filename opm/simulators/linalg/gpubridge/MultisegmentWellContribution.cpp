/*
  Copyright 2020 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h> // CMake
#include <opm/simulators/linalg/gpubridge/MultisegmentWellContribution.hpp>

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>

#if HAVE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif // HAVE_UMFPACK

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <chrono>
#include <iomanip>

extern double ctime_alloc;
extern double ctime_datatransD;
extern double ctime_wellLU;
extern double ctime_welllsD;
extern double ctime_wellBx;
extern double ctime_wellCz;

extern double ctime_gpudatatransD;
extern double ctime_gpuLU;
extern double ctime_gpulsD;
extern double ctime_gpuBx;
extern double ctime_gpuCz;

#define HIP_CALL(call)                                     \
  do {                                                     \
    hipError_t err = call;                                 \
    if (hipSuccess != err) {                               \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,  \
             hipGetErrorString(err), __FILE__, __LINE__);  \
      exit(1);                                             \
    }                                                      \
  } while (0)

#define ROCSOLVER_CALL(call)                                                   \
  do {                                                                         \
    rocblas_status err = call;                                                 \
    if (rocblas_status_success != err) {                                       \
      printf("rocSOLVER ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

  #define ROCSPARSE_CALL(call)                                                   \
  do {                                                                           \
      rocsparse_status err = call;                                               \
      if (rocsparse_status_success != err) {                                     \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n", err, __FILE__,            \
              __LINE__);                                                         \
      exit(1);                                                                   \
      }                                                                          \
  } while (0)

void checkHIPAlloc(void* ptr) {
    if (ptr == nullptr) {
        std::cerr << "HIP malloc failed." << std::endl;
        exit(1);
    }
}


namespace Opm {

template<class Scalar>
MultisegmentWellContribution<Scalar>::
MultisegmentWellContribution(unsigned int dim_, unsigned int dim_wells_,
                             unsigned int Mb_,
                             std::vector<Scalar>& Bvalues,
                             std::vector<unsigned int>& BcolIndices,
                             std::vector<unsigned int>& BrowPointers,
                             unsigned int DnumBlocks_,
                             Scalar* Dvalues,
                             UMFPackIndex* DcolPointers,
                             UMFPackIndex* DrowIndices,
                             std::vector<Scalar>& Cvalues)
:
    Mb(Mb_),                  // number of blockrows in C, D and B
    dim(dim_),                // size of blockvectors in vectors x and y, equal to MultisegmentWell::numEq
    dim_wells(dim_wells_),    // size of blocks in C, B and D, equal to MultisegmentWell::numWellEq
    M(Mb_ * dim_wells),       // number of rows, M == dim_wells*Mb
    DnumBlocks(DnumBlocks_),  // number of blocks in D
    // copy data for matrix D into vectors to prevent it going out of scope
    Cvals(std::move(Cvalues)),
    Dvals(Dvalues, Dvalues + DnumBlocks * dim_wells * dim_wells),
    Bvals(std::move(Bvalues)),
    Dcols(DcolPointers, DcolPointers + M + 1),
    Bcols(std::move(BcolIndices)),
    Drows(DrowIndices, DrowIndices + DnumBlocks * dim_wells * dim_wells),
    Brows(std::move(BrowPointers))
{
    // hipEventCreate(&startOp1);
    // hipEventCreate(&stopOp1_startOp2);
    // hipEventCreate(&stopOp2_startOp3);
    // hipEventCreate(&stopOp3);
    // hipEventCreate(&startDataTrans);
    // hipEventCreate(&stopDataTrans_startLU);
    // hipEventCreate(&stopLU);

    rocM = size(Dcols)-1;
    rocN = rocM;
    lda = rocM > rocN ? rocM : rocN;
    ldb = Mb*dim_wells;
    ipivDim = rocM > rocN ? rocN : rocM;

    Dmatrix = (double*)malloc(sizeof(double)*rocM*rocN);

    std::vector<int> signedBcols(Bcols.begin(), Bcols.end());
    std::vector<int> signedBrows(Brows.begin(), Brows.end());

    BCSRrecttoCSR(Bvals, signedBcols, signedBrows, dim_wells, dim, Bvals_, Bcols_, Brows_);

    BCSRrecttoCSR(Cvals, signedBcols, signedBrows, dim_wells, dim, Cvals_, Ccols_, Crows_);

    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSPARSE_CALL(rocsparse_create_handle(&sparse_handle));
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_B));
    ROCSPARSE_CALL(rocsparse_create_mat_info(&B_info));
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_C));
    ROCSPARSE_CALL(rocsparse_create_mat_info(&C_info));

    Dune::Timer alloc_timer;
    alloc_timer.start();
    rocSOLVERAlloc();
    alloc_timer.stop();
    ctime_alloc += alloc_timer.lastElapsed();

    // hipEventRecord(startDataTrans, 0);
    Dune::Timer dataTrans_timer;
    dataTrans_timer.start();
    matricesToDevice();
    dataTrans_timer.stop();
    ctime_datatransD += dataTrans_timer.lastElapsed();
    // hipEventRecord(stopDataTrans_startLU, 0);

    Dune::Timer LU_timer;
    LU_timer.start();
    // LU factorization
    ROCSOLVER_CALL(rocsolver_dgetrf(handle, rocM, rocN, d_Dmatrix, lda, ipiv, info));
    LU_timer.stop();
    ctime_wellLU += LU_timer.lastElapsed();
    // hipEventRecord(stopLU, 0);

    // hipEventSynchronize(stopLU);

    // hipEventElapsedTime(&time_datatrans, startDataTrans, stopDataTrans_startLU);
    // hipEventElapsedTime(&time_lu, stopDataTrans_startLU, stopLU);

    // ctime_gpudatatransD += time_datatrans/1000;
    // ctime_gpuLU += time_lu/1000;

    B_M = Brows_.size() - 1;
    B_N = *std::max_element(Bcols_.begin(), Bcols_.end()) + 1;
    B_nnz = Bvals_.size();
    C_M = Crows_.size() - 1;
    C_N = *std::max_element(Ccols_.begin(), Ccols_.end()) + 1;
    C_nnz = Cvals_.size();

    Dune::Timer Bx_timer;
    Bx_timer.start();
    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(sparse_handle, sparse_operation, B_M, B_N, B_nnz, descr_B, d_Bvals, d_Brows, d_Bcols, B_info));
    Bx_timer.stop();
    ctime_wellBx += Bx_timer.lastElapsed();

    Dune::Timer Cz_timer;
    Cz_timer.start();
    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(sparse_handle, sparse_transposition, C_M, C_N, C_nnz, descr_C, d_Cvals, d_Crows, d_Ccols, C_info));
    Cz_timer.stop();
    ctime_wellCz += Cz_timer.lastElapsed();
}

template<class Scalar>
MultisegmentWellContribution<Scalar>::~MultisegmentWellContribution()
{
    free(Dmatrix);

    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
    ROCSPARSE_CALL(rocsparse_destroy_handle(sparse_handle));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_B));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_C));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(B_info));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(C_info));

    rocSOLVERFree();
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::rocSOLVERAlloc()
{
    HIP_CALL(hipMalloc(&d_Dmatrix, sizeof(double)*rocM*rocN));
    checkHIPAlloc(d_Dmatrix);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(Cvals_)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Ccols, sizeof(double)*size(Ccols_)));
    checkHIPAlloc(d_Ccols);
    HIP_CALL(hipMalloc(&d_Crows, sizeof(double)*size(Crows_)));
    checkHIPAlloc(d_Crows);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(Bvals_)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(unsigned int)*size(Bcols_)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(unsigned int)*size(Brows_)));
    checkHIPAlloc(d_Brows);
    // HIP_CALL(hipMalloc(&d_aux_x, sizeof(double)*dim*size(Bcols_)));
    // checkHIPAlloc(d_aux_x);

    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    checkHIPAlloc(ipiv);
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
    checkHIPAlloc(info);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::matricesToDevice()
{
    HIP_CALL(hipMemcpy(d_Cvals, Cvals_.data(), size(Cvals_)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Ccols, Ccols_.data(), size(Ccols_)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Crows, Crows_.data(), size(Crows_)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, Bvals_.data(), size(Bvals_)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, Bcols_.data(), size(Bcols_)*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, Brows_.data(), size(Brows_)*sizeof(unsigned int), hipMemcpyHostToDevice));

    squareCSCtoMatrix(Dmatrix, Dvals, Drows, Dcols);
    HIP_CALL(hipMemcpy(d_Dmatrix, Dmatrix, rocM*rocN*sizeof(double), hipMemcpyHostToDevice));
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::rocSOLVERFree()
{
    HIP_CALL(hipFree(d_Dmatrix));
    HIP_CALL(hipFree(d_Cvals));
    HIP_CALL(hipFree(d_Bvals));
    HIP_CALL(hipFree(d_Bcols));
    HIP_CALL(hipFree(d_Brows));
    // HIP_CALL(hipFree(d_aux_x));

    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(info));
    HIP_CALL(hipFree(d_z));

}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::solveSystem()
{
    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix, lda, ipiv, d_z, ldb));

    // HIP_CALL(hipDeviceSynchronize());
}

// Method for operation B_w * x with rocsparse method, B_w must be in CSR format
template<class Scalar>
void MultisegmentWellContribution<Scalar>::rocsparseBx(double* vals,
                                                int* cols,
                                                int* rows,
                                                double* x,
                                                double* y) {
    alpha = 1.0;
    beta = 0.0;
    // ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(sparse_handle, sparse_operation, B_M, B_N, B_nnz, descr_B, vals, rows, cols, B_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(sparse_handle, sparse_operation, B_M, B_N, B_nnz, &alpha, descr_B, vals, rows, cols, B_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    // HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Method for operation y = y - C_w^T * x with rocsparse method, C_w must be in CSR format
template<class Scalar>
void MultisegmentWellContribution<Scalar>::rocsparseCz(double* vals,
                                                int* cols,
                                                int* rows,
                                                double* x,
                                                double* y) {
    alpha = -1.0;
    beta = 1.0;
    // ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(sparse_handle, sparse_transposition, C_M, C_N, C_nnz, descr_C, vals, rows, cols, C_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(sparse_handle, sparse_transposition, C_M, C_N, C_nnz, &alpha, descr_C, vals, rows, cols, C_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    // HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
template<class Scalar>
void MultisegmentWellContribution<Scalar>::apply(double *d_x, double *d_y)
{
    OPM_TIMEBLOCK(apply);
    HIP_CALL(hipMemset(d_z, 0.0, ldb*Nrhs*sizeof(double)));

    // hipEventRecord(startOp1, 0);
    Dune::Timer contribsCalc_timer;
    contribsCalc_timer.start();
    /**
    * d_v = d_B * d_x
    */
    rocsparseBx(d_Bvals, d_Bcols, d_Brows, d_x, d_z);
    contribsCalc_timer.stop();
    ctime_wellBx += contribsCalc_timer.lastElapsed();
    // hipEventRecord(stopOp1_startOp2, 0);
    contribsCalc_timer.start();
    /**
    * d_D * d_z = d_v
    * d_z <- d_v
    */
    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix, lda, ipiv, d_z, ldb));

    // HIP_CALL(hipDeviceSynchronize());
    contribsCalc_timer.stop();
    ctime_welllsD += contribsCalc_timer.lastElapsed();
    // hipEventRecord(stopOp2_startOp3, 0);
    contribsCalc_timer.start();
    /**
    * d_y = d_y - d_C * d_z
    */
    rocsparseCz(d_Cvals, d_Ccols, d_Crows, d_z, d_y);
    contribsCalc_timer.stop();
    ctime_wellCz += contribsCalc_timer.lastElapsed();
    // hipEventRecord(stopOp3, 0);

    // hipEventSynchronize(stopOp3);

    // hipEventElapsedTime(&time_op1, startOp1, stopOp1_startOp2);
    // hipEventElapsedTime(&time_op2, stopOp1_startOp2, stopOp2_startOp3);
    // hipEventElapsedTime(&time_op3, stopOp2_startOp3, stopOp3);

    // ctime_gpuBx += time_op1/1000;
    // ctime_gpulsD += time_op2/1000;
    // ctime_gpuCz += time_op3/1000;
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::BCSRrecttoCSR(
    std::vector<double>& Bval,
    std::vector<int>& Bcol_ind,
    std::vector<int>& Brow_ptr,
    int Br, int Bc,
    std::vector<double>& val,
    std::vector<int>& col_ind,
    std::vector<int>& row_ptr) {

    int num_br = Brow_ptr.size() - 1;
    int num_bc = *std::max_element(Bcol_ind.begin(), Bcol_ind.end()) + 1;
    int M = num_br * Br;
    int N = num_bc * Bc;

    row_ptr.resize(M + 1);
    row_ptr[0] = 0;

    int csr_idx = 0; // Current index for CSR arrays val and col_ind

    for (int I = 0; I < num_br; I++) {
        int block_start = Brow_ptr[I];
        int block_end = Brow_ptr[I + 1];

        for (int r = 0; r < Br; r++) {
            int i = I * Br + r;
            if ( i >= M) break;

            for (int block_idx = block_start; block_idx < block_end; block_idx++) {
                int J  = Bcol_ind[block_idx]; // Block column index

                for ( int c = 0; c < Bc; c++) {
                    int j = J * Bc + c; // Global column index
                    if (j >= N) continue;

                    //Assuming row-major block storage
                    int block_element_idx = block_idx * Br * Bc + r * Bc + c;

                    double value = Bval[block_element_idx];

                    if ( value != 0) {
                        val.push_back(value);
                        col_ind.push_back(j);
                        csr_idx++;
                    }
                }
            }
            if (i + 1 <= M){
                row_ptr[i + 1] = csr_idx;
            }
        }
    }
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::squareCSCtoMatrix(double *Dmatrix, std::vector<double> Dvals, std::vector<int> Drows, std::vector<int> Dcols)
{
    int lda = size(Dcols)-1;
    int nnzs = size(Dvals);

    std::vector<int> Cols(nnzs);

    for(int i=0; i<lda; i++){
      for(int j=Dcols[i];j<Dcols[i+1];j++){
        Cols[j] = i;
      }
    }

    for(int i=0; i<(lda*lda); i++){
        Dmatrix[i] = 0;
    }

    for(int i=0; i<nnzs; i++){
        Dmatrix[Drows[i]+Cols[i]*lda] = Dvals[i];
    }
}

#if HAVE_CUDA
template<class Scalar>
void MultisegmentWellContribution<Scalar>::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

template class MultisegmentWellContribution<double>;

#if FLOW_INSTANTIATE_FLOAT
template class MultisegmentWellContribution<float>;
#endif

} //namespace Opm
