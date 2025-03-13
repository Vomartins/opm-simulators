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

template<class Scalar>
__global__ void serial_blocksrmvB_x_k(const Scalar *vals,
                                 const unsigned int *cols,
                                 const unsigned int *rows,
                                 const Scalar *x,
                                 Scalar *y,
                                 const int block_dimM,
                                 const int block_dimN)
{
    const int bsM = block_dimM;
    const int bsN = block_dimN;

    const unsigned int blockRow = blockDim.x * blockIdx.x + threadIdx.x;;
    const unsigned int first_block = rows[blockRow];
    const unsigned int last_block = rows[blockRow+1];

    double local_sum;

    for (unsigned int block = first_block; block < last_block; block++){
        for (int r = 0; r < bsM; r++){
            local_sum = 0.0;
            unsigned int yidx = blockRow * bsM + r;
            for (int c = 0; c < bsN; c++){
                unsigned int Bidx = block * bsM * bsN + r * bsN + c;
                double Bvals = vals[Bidx];
                unsigned int xidx = cols[block] * bsN + c;
                double x_elem = x[xidx];
                local_sum += Bvals*x_elem;
            }
            y[yidx] += local_sum;
        }
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

                Scalar x_elem = x[xidx];

                // Perform the multiplication
                local_sum += B_elem * x_elem;
            }
        }

        // Write the result back to global memory
        y[yidx] = local_sum;
        //}
}

template <class Scalar>
__global__ void parallel_V2blocksrmvB_x_k(const Scalar *vals,
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
            Scalar x_elem = x[xidx];
            local_sum += B_elem * x_elem;
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
__global__ void reduction_operator(const unsigned int *cols,
                                        const unsigned int *rows,
                                        Scalar *x_elem,
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

            x_elem[block*bsN+c] = x[xidx];
        }
    }
}

template <class Scalar>
__global__ void parallel_V3blocksrmvB_x_k(const Scalar *vals,
                                        const unsigned int *rows,
                                        const Scalar *x_elem,
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
                unsigned int xidx = block*bsN+c;

                // Perform the multiplication
                local_sum += B_elem * x_elem[xidx];
            }
        }

        // Write the result back to global memory
        y[yidx] = local_sum;
        //}
}

template<class Scalar>
__global__ void serial_blocksrmvC_z_k(const Scalar *vals,
                                 const unsigned int *cols,
                                 const unsigned int *rows,
                                 const Scalar *z,
                                 Scalar *y,
                                 const int block_dimM,
                                 const int block_dimN)
{
    const int bsM = block_dimM;
    const int bsN = block_dimN;
    const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int blockCol = col;
    const unsigned int first_block = rows[blockCol];
    const unsigned int last_block = rows[blockCol+1];

    double local_sum;

    for (unsigned int block = first_block; block < last_block; block++){
        for (int c = 0; c < bsM; c++){
            local_sum = 0.0;
            for (int r = 0; r < bsN; r++){
                unsigned int Cidx = block * bsM * bsN + c + r * bsM;
                double Cvals = vals[Cidx];
                unsigned int zidx = blockCol * bsN + r;
                double z_elem = z[zidx];
                local_sum += Cvals*z_elem;
            }
            unsigned int yidx = cols[block] * bsM + c;
            y[yidx] -= local_sum;
        }
    }
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
__global__ void parallel_V2blocksrmvC_z_k(const Scalar *vals,
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
    z1.resize(M);
    z2.resize(M);
    ilu_info.resize(DnumBlocks);
    L_info.resize(DnumBlocks);
    U_info.resize(DnumBlocks);

    Dvals_.resize(size(Dvals));
    Dcols_.resize(size(Dvals));
    Drows_.resize(size(Dcols));

    squareCSCtoCSR(Dvals, Drows, Dcols, Dvals_, Drows_, Dcols_);

    rocM = size(Dcols)-1;
    rocN = rocM;
    lda = rocM > rocN ? rocM : rocN;
    ldb = M;

    nnzs = size(Dvals);

    Dune::Timer alloc_timer;
    alloc_timer.start();
    allocInit();
    allocCall();
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
    convertCSRtoBSR();
    analyseMatrix();
    for (unsigned int i = 0; i < DnumBlocks; ++i) {
        int block_offset = i * (dim_wells * dim_wells);
        ROCSPARSE_CALL(rocsparse_dcsrilu0_analysis(handle, \
                                dim_wells, dim_wells * dim_wells, descr_D, \
                                &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
                                ilu_info[i], rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer_D));
        std::cout << "Matrix analysed - ILU0 " <<std::endl;
        ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                                dim_wells, dim_wells * dim_wells, descr_L, \
                                &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
                                L_info[i], rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer_L));
        std::cout << "Matrix analysed - Lower " <<std::endl;
        ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                                dim_wells, dim_wells * dim_wells, descr_U, \
                                &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
                                U_info[i], rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer_U));
        std::cout << "Matrix analysed - Upper " <<std::endl;

        rocsparse_int zero_position;
        rocsparse_status status = rocsparse_csrilu0_zero_pivot(handle, ilu_info[i], &zero_position);
        if (status != rocsparse_status_success) {
            printf("--- RocSPARSE Error --- L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        }

        ROCSPARSE_CALL(rocsparse_dcsrilu0(handle, dim_wells, dim_wells * dim_wells, descr_D, \
                                &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
								ilu_info[i], rocsparse_solve_policy_auto, d_buffer_D));
    }
    LU_timer.stop();
    ctime_wellLU += LU_timer.lastElapsed();
}

MultisegmentWellContribution::~MultisegmentWellContribution()
{
    ROCSPARSE_CALL(rocsparse_destroy_handle(handle));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(bsr_descr));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_D));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_U));
    for (unsigned int i = 0; i < DnumBlocks; ++i) {
        ROCSPARSE_CALL(rocsparse_destroy_mat_info(ilu_info[i]));
        ROCSPARSE_CALL(rocsparse_destroy_mat_info(L_info[i]));
        ROCSPARSE_CALL(rocsparse_destroy_mat_info(U_info[i]));
    }
    freeInit();
    freeCall();
}

void MultisegmentWellContribution::allocInit()
{
    HIP_CALL(hipMalloc(&d_Dvals, sizeof(double)*size(Dvals_)));
    checkHIPAlloc(d_Dvals);
    HIP_CALL(hipMalloc(&d_Dcols, sizeof(rocsparse_int)*DnumBlocks));
    checkHIPAlloc(d_Dcols);
    HIP_CALL(hipMalloc(&d_Drows, sizeof(rocsparse_int)*(Mb+1)));
    checkHIPAlloc(d_Drows);
    HIP_CALL(hipMalloc(&d_Dvals_, sizeof(double)*size(Dvals_)));
    checkHIPAlloc(d_Dvals_);
    HIP_CALL(hipMalloc(&d_Dcols_, sizeof(rocsparse_int)*size(Dcols_)));
    checkHIPAlloc(d_Dcols_);
    HIP_CALL(hipMalloc(&d_Drows_, sizeof(rocsparse_int)*size(Drows_)));
    checkHIPAlloc(d_Drows_);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(Cvals)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(Bvals)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(unsigned int)*size(Bcols)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(unsigned int)*size(Brows)));
    checkHIPAlloc(d_Brows);
    HIP_CALL(hipMalloc(&d_x_elem, sizeof(double)*dim*size(Bcols)));
    checkHIPAlloc(d_x_elem);
}

void MultisegmentWellContribution::allocCall()
{
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
    HIP_CALL(hipMalloc(&d_z_aux, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z_aux);
}


void MultisegmentWellContribution::matricesToDevice()
{
    HIP_CALL(hipMemcpy(d_Dvals_, Dvals_.data(), size(Dvals_)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Dcols_, Dcols_.data(), size(Dcols_)*sizeof(rocsparse_int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Drows_, Drows_.data(), size(Drows_)*sizeof(rocsparse_int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Cvals, Cvals.data(), size(Cvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, Bvals.data(), size(Bvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, Bcols.data(), size(Bcols)*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, Brows.data(), size(Brows)*sizeof(unsigned int), hipMemcpyHostToDevice));
}

void MultisegmentWellContribution::freeInit()
{
    HIP_CALL(hipFree(d_Dvals));
    HIP_CALL(hipFree(d_Dcols));
    HIP_CALL(hipFree(d_Drows));
    HIP_CALL(hipFree(d_Dvals_));
    HIP_CALL(hipFree(d_Dcols_));
    HIP_CALL(hipFree(d_Drows_));
    HIP_CALL(hipFree(d_Cvals));
    HIP_CALL(hipFree(d_Bvals));
    HIP_CALL(hipFree(d_Bcols));
    HIP_CALL(hipFree(d_Brows));

}

void MultisegmentWellContribution::freeCall()
{
    HIP_CALL(hipFree(d_z));
    HIP_CALL(hipFree(d_z_aux));
}

void MultisegmentWellContribution::convertCSRtoBSR()
{
    ROCSPARSE_CALL(rocsparse_create_handle(&handle));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&csr_descr));
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&bsr_descr));

    ROCSPARSE_CALL(rocsparse_dcsr2bsr(handle, rocsparse_direction_row, M, M, csr_descr, d_Dvals_, d_Drows_, d_Dcols_, \
                                     dim_wells, bsr_descr, d_Dvals, d_Drows, d_Dcols));
}

void MultisegmentWellContribution::analyseMatrix()
{
    //ROCSPARSE_CALL(rocsparse_create_handle(&handle));

    // Create matrix descriptor for matrices D, L, and U
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_D));
    ROCSPARSE_CALL(rocsparse_set_mat_type(descr_D, rocsparse_matrix_type_general));
    ROCSPARSE_CALL(rocsparse_set_mat_index_base(descr_D, rocsparse_index_base_zero));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CALL(rocsparse_set_mat_type(descr_L, rocsparse_matrix_type_general));
    ROCSPARSE_CALL(rocsparse_set_mat_index_base(descr_L, rocsparse_index_base_zero));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_U));
    ROCSPARSE_CALL(rocsparse_set_mat_type(descr_U, rocsparse_matrix_type_general));
    ROCSPARSE_CALL(rocsparse_set_mat_index_base(descr_U, rocsparse_index_base_zero));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    for (unsigned int i = 0; i < DnumBlocks; ++i) {
        ROCSPARSE_CALL(rocsparse_create_mat_info(&ilu_info[i]));
        ROCSPARSE_CALL(rocsparse_create_mat_info(&L_info[i]));
        ROCSPARSE_CALL(rocsparse_create_mat_info(&U_info[i]));
    }

    // Obtain required buffer sizes
    ROCSPARSE_CALL(rocsparse_dcsrilu0_buffer_size(handle, dim_wells, dim_wells*dim_wells,
						  descr_D, d_Dvals, d_Drows, d_Dcols, ilu_info[0], &d_bufferSize_D));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, dim_wells, dim_wells*dim_wells,
						descr_L, d_Dvals, d_Drows, d_Dcols, L_info[0], &d_bufferSize_L));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, dim_wells, dim_wells*dim_wells,
						descr_U, d_Dvals, d_Drows, d_Dcols, U_info[0], &d_bufferSize_U));
    //d_bufferSize = std::max(d_bufferSize_D, std::max(d_bufferSize_L, d_bufferSize_U));
    HIP_CALL(hipMalloc(&d_buffer_D, d_bufferSize_D));
    HIP_CALL(hipMalloc(&d_buffer_L, d_bufferSize_L));
    HIP_CALL(hipMalloc(&d_buffer_U, d_bufferSize_U));
}


void MultisegmentWellContribution::solveSystem()
{
    for (unsigned int i = 0; i < DnumBlocks; ++i) {
        int block_offset = i * (dim_wells * dim_wells);
        int vec_offset = i * dim_wells;
        ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle, \
                                    operation, dim_wells, dim_wells * dim_wells, &one, descr_L, \
                                    &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
                                    L_info[i], d_z + vec_offset, d_z_aux + vec_offset, rocsparse_solve_policy_auto, d_buffer_L));
        ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle,\
                                    operation, dim_wells, dim_wells * dim_wells, &one, descr_U, \
                                    &d_Dvals[block_offset], &d_Drows[i * (dim_wells + 1)], &d_Dcols[i * dim_wells], \
                                    U_info[i], d_z_aux + vec_offset, d_z + vec_offset, rocsparse_solve_policy_auto, d_buffer_U));

    }

    HIP_CALL(hipDeviceSynchronize());
}

void MultisegmentWellContribution::serialBlocksrmvB_x(double* vals,
                                                      unsigned int* cols,
                                                      unsigned int* rows,
                                                      double* x,
                                                      double* y,
                                                      unsigned int Nbr,
                                                      int block_dimM,
                                                      int block_dimN)
{
    int Nthreads = 1;
    int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    serial_blocksrmvB_x_k<<<grid, block>>>(vals, cols, rows, x, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
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

void MultisegmentWellContribution::parallelV2BlocksrmvB_x(double* vals,
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

    parallel_V2blocksrmvB_x_k<<<grid, block, shared_memory_size>>>(vals, cols, rows, x, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::parallelV3BlocksrmvB_x(double* vals,
                                                        unsigned int* cols,
                                                        unsigned int* rows,
                                                        double* x,
                                                        double* x_elem,
                                                        double* y,
                                                        unsigned int Nbr,
                                                        int block_dimM,
                                                        int block_dimN)
{
    int Nthreads = block_dimM;
    int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    reduction_operator<<<grid, block>>>(cols, rows, x_elem, x, block_dimN);

    parallel_V3blocksrmvB_x_k<<<grid, block>>>(vals, rows, x_elem, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

void MultisegmentWellContribution::serialBlocksrmvC_z(double* vals,
                                                      unsigned int* cols,
                                                      unsigned int* rows,
                                                      double* z,
                                                      double* y,
                                                      unsigned int Nbr,
                                                      int block_dimM,
                                                      int block_dimN)
{
    int Nthreads = 1;
    int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    serial_blocksrmvC_z_k<<<grid, block>>>(vals, cols, rows, z, y, block_dimM, block_dimN);

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

void MultisegmentWellContribution::parallelV2BlocksrmvC_z(double* vals,
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
    dim3 grid(Nblocks, 1, 1);                        // One grid block per matrix block column

    size_t shared_memory_size = block_dimM * block_dimN * sizeof(double);

    parallel_V2blocksrmvC_z_k<<<grid, block, shared_memory_size>>>(vals, cols, rows, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError());      // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
void MultisegmentWellContribution::apply(double *d_x, double *d_y)
{
    OPM_TIMEBLOCK(apply);

    HIP_CALL(hipMemset(d_z, 0.0, ldb*Nrhs*sizeof(double)));

    Dune::Timer contribsCalc_timer;
    contribsCalc_timer.start();
    //serialBlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    parallelBlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    //parallelV2BlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    //parallelV3BlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_x_elem, d_z, size(Brows) - 1, dim_wells, dim);
    contribsCalc_timer.stop();
    ctime_wellBx += contribsCalc_timer.lastElapsed();
    contribsCalc_timer.start();
    solveSystem();
    contribsCalc_timer.stop();
    ctime_welllsD += contribsCalc_timer.lastElapsed();
    contribsCalc_timer.start();
    //serialBlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    parallelBlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    //parallelV2BlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    contribsCalc_timer.stop();
    ctime_wellCz += contribsCalc_timer.lastElapsed();
}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

void MultisegmentWellContribution::squareCSCtoCSR(std::vector<double> vals, std::vector<int> rows, std::vector<int> cols, std::vector<double>& vals_, std::vector<int>& rows_, std::vector<int>& cols_)
{
    int sizeDvals = size(vals);
    int sizeDcols = size(cols);

    std::vector<int> Cols(sizeDvals);

    for(int i=0; i<sizeDcols-1; i++){
      for(int j=cols[i];j<cols[i+1];j++){
        Cols[j] = i;
      }
    }

    std::vector<std::tuple<int,double,int>> ConvertVec;

    for(int i=0; i<sizeDvals; i++){
      ConvertVec.push_back(std::make_tuple(rows[i],vals[i],Cols[i]));
    }


    std::sort(ConvertVec.begin(),ConvertVec.end());

    auto it = ConvertVec.begin();
    while (it != ConvertVec.end()) {
        auto range_end = std::find_if(it, ConvertVec.end(),
            [it](const std::tuple<int, double, int>& tup) {
                return std::get<0>(tup) != std::get<0>(*it);
            });

        std::sort(it, range_end,
            [](const std::tuple<int, double, int>& a, const std::tuple<int, double, int>& b) {
                return std::get<2>(a) < std::get<2>(b);
            });

        it = range_end;
    }

    for(int i=0; i<sizeDvals; i++){
        //std::cout << "{" << std::get<0>(ConvertVec[i]) << ", "  << std::get<1>(ConvertVec[i]) << ", " << std::get<2>(ConvertVec[i]) << "}" << std::endl;
        vals_[i] = std::get<1>(ConvertVec[i]);
        cols_[i] = std::get<2>(ConvertVec[i]);
    }

    for(int target = 0; target< sizeDcols-1; target++){
      it = std::find_if(ConvertVec.begin(), ConvertVec.end(),
          [target](const std::tuple<int, double, int>& tup) {
              return std::get<0>(tup) == target;
          });

      if (it != ConvertVec.end()) {
          rows_[target] = std::distance(ConvertVec.begin(),it);
      } else {
          std::cout << target << " not found in the third element of any tuple.\n";
      }
    }

    for (int i=1; i<sizeDcols-1; i++){
        if(rows_[i] == 0){
            rows_[i] = rows_[i+1];
        }
    }
    rows_[sizeDcols-1] = cols[sizeDcols-1];
}
} //namespace Opm
