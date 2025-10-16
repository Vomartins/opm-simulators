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
#include <opm/common/TimingMacros.hpp>
#if HAVE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif // HAVE_UMFPACK

#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/bda/MultisegmentWellContribution.hpp>
#include <opm/simulators/linalg/bda/Reorder.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <chrono>
#include <iomanip>

extern double ctime_alloc;
extern double ctime_mswdatatransd;
extern double ctime_wellLU;
extern double ctime_welllsD;
extern double ctime_wellBx;
extern double ctime_wellCz;
extern double matrix_save;
extern double dmatrix_apply_count;

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

void checkHIPAlloc(void* ptr) {
    if (ptr == nullptr) {
        std::cerr << "HIP malloc failed." << std::endl;
        exit(1);
    }
}

template <class Scalar>
void saveVector(std::vector<Scalar> vec, std::string filename){
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        for (const auto& num : vec) {
            output_file << num << " ";
        }
        output_file.close();
    } else {
        std::cerr << "Error opening file." << std::endl;
    }
}

template <class Scalar>
__global__ void parallel_blocksrmvB_x_k(const Scalar *vals,
                                        const unsigned int *cols,
                                        const unsigned int *rows,
                                        const Scalar *x,
                                        Scalar *y,
                                        const int block_dimM,
                                        const int block_dimN)
{
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;

    const unsigned int blockRow = blockIdx.x;  // Each GPU block handles one block row
    const unsigned int threadRow = threadIdx.x;  // Each thread handles one row of the block
    const unsigned int first_block = rows[blockRow];
    const unsigned int last_block = rows[blockRow + 1];


    // Q: When threadRow is >= bsM?
    //if (threadRow < bsM) {
        unsigned int yidx = blockRow * bsM + threadRow;
        Scalar local_sum = 0.0;

        for (unsigned int block = first_block; block < last_block; block++) {
            for (unsigned int c = 0; c < bsN; c++) {
                unsigned int Bidx = block * bsM * bsN + threadRow * bsN + c;
                Scalar B_elem = vals[Bidx];
                unsigned int xidx = cols[block] * bsN + c;

                Scalar aux_x = x[xidx];

                // Perform the multiplication
                local_sum += B_elem * aux_x;
            }
        }

        // Write the result back to global memory
        y[yidx] = local_sum;
}

template <class Scalar>
__global__ void parallel_V1blocksrmvB_x_k(const Scalar *vals,
                                          const unsigned int *cols,
                                          const unsigned int *rows,
                                          const Scalar *x,
                                          Scalar *y,
                                          const int block_dimM,
                                          const int block_dimN) {
    const unsigned int bsM = block_dimM; // Block row size
    const unsigned int bsN = block_dimN; // Block column size
    const unsigned int blockRow = blockIdx.x; // Block row index
    const unsigned int threadRow = threadIdx.y; // Thread's column index within block
    const unsigned int waveRow = threadIdx.x; // Wave row index

    // Shared memory to store intermediate values
    extern __shared__ Scalar shared_data[]; // Size should be bsM * bsN
    unsigned int shared_idx = waveRow * bsN + threadRow;

    const unsigned int first_block = rows[blockRow];
    const unsigned int last_block = rows[blockRow + 1];

    if (last_block > first_block) {
        unsigned int yidx = blockRow * bsM + waveRow; // Global memory index for y
        Scalar local_sum = 0.0;

        // Compute thread-level scalar multiplication
        for (unsigned int block = first_block; block < last_block; block++) {
            unsigned int Bidx = block * bsM * bsN + waveRow * bsN + threadRow;
            Scalar B_elem = vals[Bidx];
            unsigned int xidx = cols[block] * bsN + threadRow;
            Scalar aux_x = x[xidx];
            local_sum += B_elem * aux_x;
        }

        // Store the result in shared memory
        shared_data[shared_idx] = local_sum;
        __syncthreads();

        // Perform reduction within shared memory
        for (unsigned int offset = (bsN + 1) / 2; offset > 0; offset /= 2) {
            if (threadRow < offset && threadRow + offset < bsN) {
                shared_data[shared_idx] += shared_data[shared_idx + offset];
            }
            //__syncthreads();
        }

        // Only the first thread in the wave writes the final result to global memory
        if (threadRow == 0) {
            y[yidx] = shared_data[shared_idx];
        }
    }
}

template <class Scalar>
__global__ void contiguous_operator(const unsigned int *cols,
                                        const unsigned int *rows,
                                        Scalar *aux_x,
                                        const Scalar *x,
                                        const int block_dimN)
{
    const unsigned int bsN = block_dimN;

    const unsigned int blockRow = blockIdx.x;  // Each GPU block handles one block row
    const unsigned int first_block = rows[blockRow];
    const unsigned int last_block = rows[blockRow + 1];

    for (unsigned int block = first_block; block < last_block; block++) {
        for (unsigned int c = 0; c < bsN; c++) {
            unsigned int xidx = cols[block] * bsN + c;
            aux_x[block*bsN+c] = x[xidx];
        }
    }
}

template <class Scalar>
__global__ void parallel_V2blocksrmvB_x_k(const Scalar *vals,
                                        const unsigned int *rows,
                                        const Scalar *aux_x,
                                        Scalar *y,
                                        const int block_dimM,
                                        const int block_dimN)
{
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;

    const unsigned int blockRow = blockIdx.x;  // Each block handles one block row of C
    const unsigned int threadRow = threadIdx.x;  // Thread's row index within block (r)
    //const unsigned int waveRow = threadIdx.y; // Thread's column index within block (c)
    const unsigned int first_block = rows[blockRow];
    const unsigned int last_block = rows[blockRow + 1];

    // Q: When threadRow is >= bsM?
    //if (threadRow < bsM) {
        unsigned int yidx = blockRow * bsM + threadRow;
        Scalar local_sum = 0.0;

        for (unsigned int block = first_block; block < last_block; block++) {
            for (unsigned int c = 0; c < bsN; c++){
                unsigned int Bidx = block * bsM * bsN + threadRow * bsN + c;
                Scalar B_elem = vals[Bidx];
                unsigned int xidx = block*bsN + c;

                // Perform the multiplication
                local_sum += B_elem * aux_x[xidx];
            }
        }

        // Write the result back to global memory
        y[yidx] = local_sum;
        //}
}

template<class Scalar>
__global__ void parallel_blocksrmvC_z_k(const Scalar *vals,
                                        const unsigned int *cols,
                                        const unsigned int *rows,
                                        const Scalar *z,
                                        Scalar *y,
                                        const int block_dimM,
                                        const int block_dimN)
{
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;

    const unsigned int blockCol = blockIdx.x;
    const unsigned int threadCol = threadIdx.x;
    const unsigned int first_block = rows[blockCol];
    const unsigned int last_block = rows[blockCol+1];

    if (threadCol < bsM) {
        for (unsigned int block = first_block; block < last_block; block++){
            Scalar local_sum = 0.0;
            unsigned int yidx = cols[block] * bsM + threadCol;
            for (unsigned int r = 0; r < bsN; r++){
                unsigned int Cidx = block * bsM * bsN + threadCol + r * bsM;
                Scalar Cvals = vals[Cidx];
                unsigned int zidx = blockCol * bsN + r;

                Scalar z_elem = z[zidx];

                local_sum += Cvals*z_elem;
            }
            y[yidx] -= local_sum;
        }
    }
}

template <class Scalar>
__global__ void parallel_V1blocksrmvC_z_k(const Scalar *vals,
                                        const unsigned int *cols,
                                        const unsigned int *rows,
                                        const Scalar *z,
                                        Scalar *y,
                                        const int block_dimM,
                                        const int block_dimN)
{
    const int bsM = block_dimM;
    const int bsN = block_dimN;

    const unsigned int blockCol = blockIdx.x; // Each block handles one block column of C
    const unsigned int threadCol = threadIdx.y; // Thread's column index within block (r)
    const unsigned int waveCol = threadIdx.x; // Thread's row index within block (c)

    const unsigned int first_block = rows[blockCol];
    const unsigned int last_block = rows[blockCol + 1];

    if (last_block > first_block) {
        // Each thread computes its contribution for one element of y
        for (unsigned int block = first_block; block < last_block; block++) {
            // Compute Cidx: block * bsM * bsN + c + r * bsM
            unsigned int Cidx = block * bsM * bsN + waveCol + threadCol * bsM;
            Scalar C_elem = vals[Cidx];

            // Compute zidx: blockCol * bsN + r
            unsigned int zidx = blockCol * bsN + threadCol;
            Scalar z_elem = z[zidx];

            // Compute contribution: Cvals * z_elem
            Scalar contribution = C_elem * z_elem;

            // Compute yidx: cols[block] * bsM + c
            unsigned int yidx = cols[block] * bsM + waveCol;

            // Update y[yidx] atomically (to handle overlapping updates)
            atomicAdd(&y[yidx], -contribution);
        }
    }
}

// TODO: Create parallel_V2blocksrmvC_z_k with contiguous operator

namespace Opm
{

MultisegmentWellContribution::MultisegmentWellContribution(unsigned int dim_, unsigned int dim_wells_,
        unsigned int Mb_,
        std::vector<double> &Bvalues, std::vector<unsigned int> &BcolIndices, std::vector<unsigned int> &BrowPointers,
        unsigned int DnumBlocks_, double *Dvalues, UMFPackIndex *DcolPointers, UMFPackIndex *DrowIndices,
        std::vector<double> &Cvalues)
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

    rocM = size(Dcols)-1;
    rocN = rocM;
    lda = rocM > rocN ? rocM : rocN;
    ldb = Mb*dim_wells;
    ipivDim = rocM > rocN ? rocN : rocM;

    //std::cout << "=== === === rocM: " << rocM << " ldb: " << ldb << " === === === " << std::endl;

    Dmatrix = (double*)malloc(sizeof(double)*rocM*rocN);

    // saveVector(Cvals, "Cvals.txt");
    // saveVector(Bcols, "Bcols.txt");
    // saveVector(Brows, "Brows.txt");
    // std::exit(0);

    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    Dune::Timer alloc_timer;
    alloc_timer.start();
    rocSOLVERAlloc();
    alloc_timer.stop();
    ctime_alloc += alloc_timer.lastElapsed();

    Dune::Timer dataTrans_timer;
    dataTrans_timer.start();
    matricesToDevice();
    dataTrans_timer.stop();
    ctime_mswdatatransd += dataTrans_timer.lastElapsed();

    Dune::Timer LU_timer;
    LU_timer.start();
    // LU factorization
    ROCSOLVER_CALL(rocsolver_dgetrf(handle, rocM, rocN, d_Dmatrix, lda, ipiv, info));
    LU_timer.stop();
    ctime_wellLU += LU_timer.lastElapsed();
}

MultisegmentWellContribution::~MultisegmentWellContribution()
{
    free(Dmatrix);

    ROCSOLVER_CALL(rocblas_destroy_handle(handle));

    rocSOLVERFree();
}

void MultisegmentWellContribution::rocSOLVERAlloc()
{
    HIP_CALL(hipMalloc(&d_Dmatrix, sizeof(double)*rocM*rocN));
    checkHIPAlloc(d_Dmatrix);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(Cvals)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(Bvals)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(unsigned int)*size(Bcols)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(unsigned int)*size(Brows)));
    checkHIPAlloc(d_Brows);
    HIP_CALL(hipMalloc(&d_aux_x, sizeof(double)*dim*size(Bcols)));
    checkHIPAlloc(d_aux_x);

    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    checkHIPAlloc(ipiv);
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
    checkHIPAlloc(info);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
}

void MultisegmentWellContribution::matricesToDevice()
{
    HIP_CALL(hipMemcpy(d_Cvals, Cvals.data(), size(Cvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, Bvals.data(), size(Bvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, Bcols.data(), size(Bcols)*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, Brows.data(), size(Brows)*sizeof(unsigned int), hipMemcpyHostToDevice));

    Accelerator::squareCSCtoMatrix(Dmatrix, Dvals, Drows, Dcols);
    HIP_CALL(hipMemcpy(d_Dmatrix, Dmatrix, rocM*rocN*sizeof(double), hipMemcpyHostToDevice));
}

void MultisegmentWellContribution::rocSOLVERFree()
{
    HIP_CALL(hipFree(d_Dmatrix));
    HIP_CALL(hipFree(d_Cvals));
    HIP_CALL(hipFree(d_Bvals));
    HIP_CALL(hipFree(d_Bcols));
    HIP_CALL(hipFree(d_Brows));
    HIP_CALL(hipFree(d_aux_x));

    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(info));
    HIP_CALL(hipFree(d_z));

}

void MultisegmentWellContribution::solveSystem()
{
    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix, lda, ipiv, d_z, ldb));

    HIP_CALL(hipDeviceSynchronize());
}

void MultisegmentWellContribution::parallelBlocksrmvB_x(double* vals,
                                                        unsigned int* cols,
                                                        unsigned int* rows,
                                                        double* x,
                                                        double* y,
                                                        unsigned int Nbr,
                                                        int block_dimM,
                                                        int block_dimN)
{
    int Nthreads = block_dimM;
    int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    parallel_blocksrmvB_x_k<<<grid, block>>>(vals, cols, rows, x, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::parallelV1BlocksrmvB_x(double* vals,
                                                      unsigned int* cols,
                                                      unsigned int* rows,
                                                      double* x,
                                                      double* y,
                                                      unsigned int Nbr,
                                                      int block_dimM,
                                                      int block_dimN) {
    // int Nthreads = block_dimM * block_dimN; // Number of threads per block
    int Nblocks = Nbr; // Number of blocks
    size_t shared_memory_size = block_dimM * block_dimN * sizeof(double);

    dim3 block(block_dimM, block_dimN, 1);
    dim3 grid(Nblocks, 1, 1);

    parallel_V1blocksrmvB_x_k<<<grid, block, shared_memory_size>>>(vals, cols, rows, x, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::parallelV2BlocksrmvB_x(double* vals,
                                                        unsigned int* cols,
                                                        unsigned int* rows,
                                                        double* x,
                                                        double* aux_x,
                                                        double* y,
                                                        unsigned int Nbr,
                                                        int block_dimM,
                                                        int block_dimN)
{
    int Nthreads = block_dimM;
    int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    contiguous_operator<<<grid, block>>>(cols, rows, aux_x, x, block_dimN);

    parallel_V2blocksrmvB_x_k<<<grid, block>>>(vals, rows, aux_x, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::parallelBlocksrmvC_z(double* vals,
                                                        unsigned int* cols,
                                                        unsigned int* rows,
                                                        double* z,
                                                        double* y,
                                                        unsigned int Nbr,
                                                        int block_dimM,
                                                        int block_dimN)
{
    int Nthreads = block_dimM; // Threads per block
    int Nblocks = Nbr;
    dim3 block(Nthreads, 1 ,1);                      // One thread block per block column
    dim3 grid(Nblocks, 1, 1);                        // One grid block per matrix block column

    parallel_blocksrmvC_z_k<<<grid, block>>>(vals, cols, rows, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError());      // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::parallelV1BlocksrmvC_z(double* vals,
                                                        unsigned int* cols,
                                                        unsigned int* rows,
                                                        double* z,
                                                        double* y,
                                                        unsigned int Nbr,
                                                        int block_dimM,
                                                        int block_dimN)
{
    // int Nthreads = block_dimM*block_dimN; // Threads per block
    int Nblocks = Nbr;
    dim3 block(block_dimM, block_dimN ,1);                      // One thread block per block column
    dim3 grid(Nblocks, 1, 1);                                   // One grid block per matrix block column

    size_t shared_memory_size = block_dimM * block_dimN * sizeof(double);

    parallel_V1blocksrmvC_z_k<<<grid, block, shared_memory_size>>>(vals, cols, rows, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError());      // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Method for operation B_w * x with rocsparse method, B_w must be in CSR format
void MultisegmentWellContribution::rocsparseBx(double* vals,
                                                int* cols,
                                                int* rows,
                                                double* x,
                                                double* y) {
    int B_M = Brows_.size() - 1;
    int B_N = *std::max_element(Bcols_.begin(), Bcols_.end()) + 1;
    int B_nnz = Bvals_.size();
    double alpha = 1.0;
    double beta = 0.0;

    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_none, B_M, B_N, B_nnz, descr_B, vals, rows, cols, B_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_none, B_M, B_N, B_nnz, &alpha, descr_B, vals, rows, cols, B_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Method for operation y = y - C_w^T * x with rocsparse method, C_w must be in CSR format
void MultisegmentWellContribution::rocsparseCz(double* vals,
                                                int* cols,
                                                int* rows,
                                                double* x,
                                                double* y) {
    int C_M = Crows_.size() - 1;
    int C_N = *std::max_element(Ccols_.begin(), Ccols_.end()) + 1;
    int C_nnz = Cvals_.size();
    double alpha = -1.0;
    double beta = 1.0;

    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, descr_C, vals, rows, cols, C_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, &alpha, descr_C, vals, rows, cols, C_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

/**
* @brief Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
* @brief y -= (C^T * (D^-1 * (B * x)))
*/
void MultisegmentWellContribution::apply(double *d_x, double *d_y)
{
    // saveVector(Cvals, "Cvals.txt");
    // saveVector(Bvals, "Bvals.txt");
    // saveVector(Bcols, "Bcols.txt");
    // saveVector(Brows, "Brows.txt");
    // saveVector(Dvals, "Dvals.txt");
    // saveVector(Dcols, "Dcols.txt");
    // saveVector(Drows, "Drows.txt");
    // std::exit(0);

    OPM_TIMEBLOCK(apply);

    HIP_CALL(hipMemset(d_z, 0.0, ldb*Nrhs*sizeof(double)));

    Dune::Timer contribsCalc_timer;
    contribsCalc_timer.start();
    /**
    * d_v = d_B * d_x
    */
    //parallelBlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    //parallelV1BlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    // parallelV2BlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_aux_x, d_z, size(Brows) - 1, dim_wells, dim);
    rocsparseBx(d_Bvals, d_Bcols, d_Brows, d_x, d_z);
    contribsCalc_timer.stop();
    ctime_wellBx += contribsCalc_timer.lastElapsed();
    contribsCalc_timer.start();
    /**
    * d_D * d_z = d_v
    * d_z <- d_v
    */
    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix, lda, ipiv, d_z, ldb));

    HIP_CALL(hipDeviceSynchronize());
    contribsCalc_timer.stop();
    ctime_welllsD += contribsCalc_timer.lastElapsed();
    contribsCalc_timer.start();
    /**
    * d_y = d_y - d_C * d_z
    */
    // parallelBlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    //parallelV1BlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    rocsparseCz(d_Cvals, d_Ccols, d_Crows, d_z, d_y);
    contribsCalc_timer.stop();
    ctime_wellCz += contribsCalc_timer.lastElapsed();
}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

} //namespace Opm
