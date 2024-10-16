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

#include <opm/simulators/linalg/bda/MultisegmentWellContribution.hpp>
#include <opm/simulators/linalg/bda/Reorder.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>

extern double well_systems;
extern double well_counter;
extern double vec_counter;
extern double ctime_mswdatatrans;
extern double ctime_mswdatatransd;
extern double ctime_welllsD;
extern double ctime_alloc;
extern double ctime_rocsoldatatrans;

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

template<class Scalar>
__global__ void blocksrmvBx_k(const Scalar *vals,
                                   const unsigned int *cols,
                                   const unsigned int *rows,
                                   const unsigned int Nb,
                                   const Scalar *x,
                                   const Scalar *rhs,
                                   Scalar *out,
                                   const unsigned int block_dimM,
                                   const unsigned int block_dimN,
                                   const double op_sign)
{
    extern __shared__ Scalar tmp[];
    const unsigned int warpsize = warpSize;
    const unsigned int bsize = blockDim.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx_b = gid / bsize;
    const unsigned int idx_t = threadIdx.x;
    unsigned int idx = idx_b * bsize + idx_t;
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;
    const unsigned int num_active_threads = (warpsize/bsM/bsN)*bsM*bsN;
    const unsigned int num_blocks_per_warp = warpsize/bsM/bsN;
    const unsigned int NUM_THREADS = gridDim.x;
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    unsigned int target_block_row = idx / warpsize;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bsM) % bsN;
    const unsigned int r = lane % bsM;

    // for 3x3 blocks:
    // num_active_threads: 27 (CUDA) vs 63 (ROCM)
    // num_blocks_per_warp: 3 (CUDA) vs  7 (ROCM)
    unsigned int offsetTarget = warpsize == 64 ? 48 : 32;

    while(target_block_row < Nb){
        unsigned int first_block = rows[target_block_row];
        unsigned int last_block = rows[target_block_row+1];
        unsigned int block = first_block + lane / (bsM*bsN);
        Scalar local_out = 0.0;

        if(lane < num_active_threads){
            for(; block < last_block; block += num_blocks_per_warp){
                Scalar x_elem = x[cols[block]*bsN + c];
                Scalar A_elem = vals[block*bsM*bsN + c + r*bsN];
                local_out += x_elem * A_elem;
            }
        }

        // do reduction in shared mem
        tmp[lane] = local_out;

        for(unsigned int offset = block_dimM; offset <= offsetTarget; offset <<= 1)
        {
            if (lane + offset < warpsize)
            {
                tmp[lane] += tmp[lane + offset];
            }
            __syncthreads();
        }

        if(lane < bsM){
            unsigned int row = target_block_row*bsM + lane;
            out[row] = rhs[row] + op_sign*tmp[lane];
        }
        target_block_row += num_warps_in_grid;
    }
}
/*
template<class Scalar>
__global__ void blocksrmvCtz_k(const Scalar *vals,
                                   const unsigned int *cols,
                                   const unsigned int *rows,
                                   const unsigned int Nb,
                                   const Scalar *x,
                                   const Scalar *rhs,
                                   Scalar *out,
                                   const unsigned int block_dimM,
                                   const unsigned int block_dimN,
                                   const double op_sign)
{
    extern __shared__ Scalar tmp[];
    const unsigned int warpsize = warpSize;
    const unsigned int bsize = blockDim.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx_b = gid / bsize;
    const unsigned int idx_t = threadIdx.x;
    unsigned int idx = idx_b * bsize + idx_t;
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;
    const unsigned int num_active_threads = (warpsize/bsM/bsN)*bsM*bsN;
    const unsigned int num_blocks_per_warp = warpsize/bsM/bsN;
    const unsigned int NUM_THREADS = gridDim.x;
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    unsigned int target_block_row = idx / warpsize;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bsN) % bsM;
    const unsigned int r = lane % bsN;

    // for 3x3 blocks:
    // num_active_threads: 27 (CUDA) vs 63 (ROCM)
    // num_blocks_per_warp: 3 (CUDA) vs  7 (ROCM)
    unsigned int offsetTarget = warpsize == 64 ? 48 : 32;

    while(target_block_row < Nb){
        unsigned int first_block = rows[target_block_row];
        unsigned int last_block = rows[target_block_row+1];
        unsigned int block = first_block + lane / (bsM*bsN);
        Scalar local_out = 0.0;

        if(lane < num_active_threads){
            for(; block < last_block; block += num_blocks_per_warp){
                Scalar x_elem = x[target_block_row*bsM + c];
                Scalar A_elem = vals[block*bsM*bsN + c*bsN + r];
                local_out += x_elem * A_elem;
            }
        }
        // do reduction in shared mem
        tmp[lane] = local_out;

        for(unsigned int offset = block_dimN; offset <= offsetTarget; offset <<= 1)
        {
            if (lane + offset < warpsize)
            {
                tmp[lane] += tmp[lane + offset];
            }
            __syncthreads();
        }

        if(lane < bsN){
            unsigned int row = cols[block]*bsN + lane;
            out[row] = rhs[row] + op_sign*tmp[lane];
        }
        target_block_row += num_warps_in_grid;
    }
}
*/
template<class Scalar>
__global__ void blocksrmvC_z_k(const Scalar *vals,
                                 const unsigned int *cols,
                                 const unsigned int *rows,
                                 const unsigned int Nb,
                                 const Scalar *z,
                                 Scalar *y,
                                 const unsigned int block_dimM,
                                 const unsigned int block_dimN)
{
    extern __shared__ Scalar tmp[];
    const unsigned int warpsize = warpSize;
    const unsigned int bsize = blockDim.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx_b = gid / bsize;
    const unsigned int idx_t = threadIdx.x;
    unsigned int idx = idx_b * bsize + idx_t;
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;
    const unsigned int num_active_threads = (warpsize / bsM / bsN) * bsM * bsN;
    const unsigned int num_blocks_per_warp = warpsize / bsM / bsN;
    unsigned int target_block_row = idx / warpsize;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = lane % bsM;  // Access the row in C
    const unsigned int r = lane / bsM;   // Access the column in C

    while (target_block_row < Nb) {
        unsigned int first_block = rows[target_block_row];
        unsigned int last_block = rows[target_block_row + 1];
        unsigned int block = first_block + lane / (bsM * bsN);
        Scalar local_out = 0.0;

        // Compute Cz
        if (lane < num_active_threads) {
            for (; block < last_block; block += num_blocks_per_warp) {
                Scalar z_elem = z[cols[block] * bsN + r];  // Access z using the column of the current block
                Scalar A_elem = vals[block * bsM * bsN + c * bsN + r];  // Access corresponding element of C
                local_out += A_elem * z_elem;  // Accumulate
            }
        }

        // Store the result in shared memory
        tmp[lane] = local_out;

        // Perform reduction to sum up the results
        for (unsigned int offset = block_dimN; offset > 0; offset >>= 1) {
            if (lane < offset) {
                tmp[lane] += tmp[lane + offset];
            }
            __syncthreads();
        }

        // Perform the final subtraction and update y
        if (lane < bsM) {
            unsigned int row = target_block_row * bsM + lane; // Calculate the row index in y
            y[row] -= tmp[0];  // Update y: y = y - Cz
        }

        target_block_row += (warpsize / blockDim.x);
    }
}

namespace Opm
{

MultisegmentWellContribution::MultisegmentWellContribution(unsigned int dim_, unsigned int dim_wells_,
        unsigned int Mb_,
        std::vector<double> &Bvalues, std::vector<unsigned int> &BcolIndices, std::vector<unsigned int> &BrowPointers,
        unsigned int DnumBlocks_, double *Dvalues, UMFPackIndex *DcolPointers, UMFPackIndex *DrowIndices,
        std::vector<double> &Cvalues, int matrixDtrans)
    :
    dim(dim_),                // size of blockvectors in vectors x and y, equal to MultisegmentWell::numEq
    dim_wells(dim_wells_),    // size of blocks in C, B and D, equal to MultisegmentWell::numWellEq
    M(Mb_ * dim_wells),       // number of rows, M == dim_wells*Mb
    Mb(Mb_),                  // number of blockrows in C, D and B
    DnumBlocks(DnumBlocks_),  // number of blocks in D
    // copy data for matrix D into vectors to prevent it going out of scope
    Dvals(Dvalues, Dvalues + DnumBlocks * dim_wells * dim_wells),
    Dcols(DcolPointers, DcolPointers + M + 1),
    Drows(DrowIndices, DrowIndices + DnumBlocks * dim_wells * dim_wells)
{
    Cvals = std::move(Cvalues);
    Bvals = std::move(Bvalues);
    Bcols = std::move(BcolIndices);
    Brows = std::move(BrowPointers);

    rocM = size(Dcols)-1;
    rocN = rocM;
    lda = rocM > rocN ? rocM : rocN;
    ldb = Mb*dim_wells;
    ipivDim = rocM > rocN ? rocN : rocM;

    matrixDtransfer = matrixDtrans;

    z1.resize(Mb * dim_wells);
    z2.resize(Mb * dim_wells);

    //umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    //umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
}

MultisegmentWellContribution::~MultisegmentWellContribution()
{
    //umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    //umfpack_di_free_numeric(&UMFPACK_Numeric);

}

void MultisegmentWellContribution::hipAlloc()
{
    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    HIP_CALL(hipMalloc(&d_Dmatrix_hip, sizeof(double)*rocM*rocN));
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
    HIP_CALL(hipMalloc(&z_hip, sizeof(double)*ldb*Nrhs));
    HIP_CALL(hipMalloc(&rhs_hip, sizeof(double)*ldb*Nrhs));
    HIP_CALL(hipMalloc(&d_Cvals_hip, sizeof(double)*size(Cvals)));
    HIP_CALL(hipMalloc(&d_Bvals_hip, sizeof(double)*size(Bvals)));
    HIP_CALL(hipMalloc(&d_Bcols_hip, sizeof(unsigned int)*size(Bcols)));
    HIP_CALL(hipMalloc(&d_Brows_hip, sizeof(unsigned int)*size(Brows)));
}

void MultisegmentWellContribution::matricesToDevice()
{
    double* Dmatrix = Accelerator::squareCSCtoMatrix(Dvals, Drows, Dcols);

    HIP_CALL(hipMemcpy(d_Dmatrix_hip, Dmatrix, rocM*rocN*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Cvals_hip, Cvals.data(), size(Cvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals_hip, Bvals.data(), size(Bvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols_hip, Bcols.data(), size(Bcols)*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows_hip, Brows.data(), size(Brows)*sizeof(unsigned int), hipMemcpyHostToDevice));
    std::vector<double> rhs(ldb*Nrhs, 0.0);
    HIP_CALL(hipMemcpy(rhs_hip, rhs.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
}

void MultisegmentWellContribution::freeRocSOLVER()
{
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(d_Dmatrix_hip));
    HIP_CALL(hipFree(z_hip));
    HIP_CALL(hipFree(d_Cvals_hip));
    HIP_CALL(hipFree(d_Bvals_hip));
    HIP_CALL(hipFree(d_Bcols_hip));
    HIP_CALL(hipFree(d_Brows_hip));
    HIP_CALL(hipFree(rhs_hip));
    HIP_CALL(hipFree(info));
    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
}

void MultisegmentWellContribution::solveSystem()
{
    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, rocM, rocN, d_Dmatrix_hip, lda, ipiv, info));

    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix_hip, lda, ipiv, z_hip, ldb));
}

void MultisegmentWellContribution::blocksrmvBx(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* rhs, double* out, unsigned int Nb, unsigned int block_dimM, unsigned int block_dimN, const double op_sign)
{
  unsigned int blockDim = 32;
  unsigned int number_wg = std::ceil(Nb/blockDim);
  unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
  unsigned int gridDim = num_work_groups*blockDim;
  unsigned int shared_mem_size = blockDim*sizeof(double)* block_dimM * block_dimN;

  blocksrmvBx_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size>>>(vals, cols, rows, Nb, x, rhs, out, block_dimM, block_dimN, op_sign);

  HIP_CALL(hipGetLastError()); // Check for errors
  HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}
/*
void MultisegmentWellContribution::blocksrmvCtz(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* rhs, double* out, unsigned int Nb, unsigned int block_dimM, unsigned int block_dimN, const double op_sign)
{
  unsigned int blockDim = 32;
  unsigned int number_wg = std::ceil(Nb/blockDim);
  unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
  unsigned int gridDim = num_work_groups*blockDim;
  unsigned int shared_mem_size = blockDim*sizeof(double)* block_dimM * block_dimN;

  blocksrmvCtz_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size>>>(vals, cols, rows, Nb, x, rhs, out, block_dimM, block_dimN, op_sign);

  HIP_CALL(hipGetLastError()); // Check for errors
  HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}
*/
void MultisegmentWellContribution::blocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nb, unsigned int block_dimM, unsigned int block_dimN)
{
    unsigned int blockDim = 32;  // Set the block size
    //unsigned int number_wg = std::ceil(Nb/blockDim);
    //unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
    //unsigned int gridDim = num_work_groups*blockDim;
    unsigned int gridDim = (Nb + blockDim - 1) / blockDim;  // Calculate grid size
    unsigned int shared_mem_size = blockDim * sizeof(double) * block_dimM * block_dimN;  // Allocate shared memory size

    blocksrmvC_z_k<<<gridDim, blockDim, shared_mem_size>>>(vals, cols, rows, Nb, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Uncomment for synchronization if needed
}

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
void MultisegmentWellContribution::apply(double *d_x, double *d_y/*, double *h_x, double *h_y*/)
{
    Dune::Timer alloc_timer;
    alloc_timer.start();
    hipAlloc();
    alloc_timer.stop();
    ctime_alloc += alloc_timer.lastElapsed();

    Dune::Timer dataTransD_timer;
    dataTransD_timer.start();
    matricesToDevice();
    dataTransD_timer.stop();
    ctime_mswdatatransd += dataTransD_timer.lastElapsed();

    OPM_TIMEBLOCK(apply);
    // reset z1 and z2
    /*
    std::fill(z1.begin(), z1.end(), 0.0);
    std::fill(z2.begin(), z2.end(), 0.0);

    // z1 = B * x
    for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k) {
                    temp += Bvals[blockID * dim * dim_wells + j * dim + k] * h_x[colIdx * dim + k];
                }
                z1[row * dim_wells + j] += temp;
            }
        }
    }
    */

    Dune::Timer dataTransWell_timer;
    /*
    dataTransWell_timer.start();
    HIP_CALL(hipMemcpy(z_hip, z1.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
    dataTransWell_timer.stop();
    ctime_rocsoldatatrans += dataTransWell_timer.lastElapsed();
    */
    Dune::Timer linearSysD_timer;
    linearSysD_timer.start();
    blocksrmvBx(d_Bvals_hip, d_Bcols_hip, d_Brows_hip, d_x, rhs_hip, z_hip, size(Brows)-1, dim_wells, dim, +1.0);
    solveSystem();
    std::cout << "Linear system solved!" << std::endl;
    blocksrmvC_z(d_Cvals_hip, d_Bcols_hip, d_Brows_hip, z_hip, d_y, size(Brows) - 1, dim_wells, dim);
    //HIP_CALL(hipDeviceSynchronize());
    linearSysD_timer.stop();
    ctime_welllsD += linearSysD_timer.lastElapsed();
/*
    dataTransWell_timer.start();
    HIP_CALL(hipMemcpy(z2.data(), z_hip, rocM*sizeof(double),hipMemcpyDeviceToHost));
    dataTransWell_timer.stop();
    ctime_rocsoldatatrans += dataTransWell_timer.lastElapsed();
*/
    freeRocSOLVER();

    // z2 = D^-1 * (B * x)
    // umfpack
    /*
    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);
    std::cout << "[ ";
    for (const auto& val : z2) std::cout << val << " ";
    std::cout << "]";
    std::cout << std::endl;
    std::cout << std::endl;
    */
/*
    // y -= (C^T * z2)
    // y -= (C^T * (D^-1 * (B * x)))
    for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim_wells; ++k) {
                    temp += Cvals[blockID * dim * dim_wells + j + k * dim] * z2[row * dim_wells + k];
                }
                h_y[colIdx * dim + j] -= temp;
            }
        }
    }
*/
}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

} //namespace Opm

