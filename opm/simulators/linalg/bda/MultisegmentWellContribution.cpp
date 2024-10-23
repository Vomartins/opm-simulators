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

extern double ctime_alloc;
extern double ctime_mswdatatransd;
extern double ctime_welllsD;

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
                                   const unsigned int Nbr,
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

    while(target_block_row < Nbr){
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
__global__ void blocksrmvC_z_k(const Scalar *vals,
                                 const unsigned int *cols,
                                 const unsigned int *rows,
                                 const unsigned int Nb,
                                 const unsigned int Nbr,
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

    //unsigned int offsetTarget = warpsize == 64 ? 32 : 16;

    while (target_block_row < Nbr) {
        unsigned int first_block = rows[target_block_row];
        unsigned int last_block = rows[target_block_row + 1];
        unsigned int block = first_block + lane / (bsM * bsN);
        //Scalar local_out = 0.0;

        // Compute Cz
        //if (lane < num_active_threads) {
            for (; block < last_block; block += num_blocks_per_warp) {
                Scalar z_elem = z[target_block_row*bsN + r];  // Access z using the column of the current block
                Scalar A_elem = vals[block * bsM * bsN + c + r*bsM]; // Access corresponding element of C
                //local_out += A_elem * z_elem; // Accumulate
                unsigned int row = cols[block] * bsM + c;
                y[row] -= A_elem * z_elem;
            }
        //}


        // Store the result in shared memory
        tmp[lane] = local_out;

        // Perform reduction to sum up the results
        for (unsigned int offset = block_dimN; offset > 0; offset >>= 1) {
            if (lane < offset) {
                tmp[lane] += tmp[lane + offset];
            }
            __syncthreads();
        }

        // for(unsigned int offset = bsN; offset <= offsetTarget; offset <<= 1)
        // {
        //    if (lane + offset < warpsize)
        //    {
        //        tmp[lane] += tmp[lane + offset];
        //    }
        //    __syncthreads();
        // }


        // Perform the final subtraction and update y
        if (block < Nb) {
            unsigned int row = cols[block] * bsM + c; // Calculate the row index in y
            //printf("block %u, col %u, row %i, target_block_row: %i\n", block ,cols[block], row, target_block_row);
            y[row] -= tmp[0];  // Update y: y = y - Cz
        }

        target_block_row += (warpsize / blockDim.x);

    }
}
*/

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

    const unsigned int blockCol = col / blockDim.x;
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
                //printf("Cvals: %.12f(%u), z_elem: %.12f(%u), local_sum: %.12f\n", Cvals, Cidx, z_elem, zidx, local_sum);
            }
            unsigned int yidx = cols[block] * bsM + c;
            y[yidx] -= local_sum;
            //printf("y_elem: %.12f(%u)\n", y[yidx], yidx);
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

    rhs.resize(ldb*Nrhs, 0.0);

    z1.resize(Mb * dim_wells);
    z2.resize(Mb * dim_wells);

    Dmatrix = (double*)malloc(sizeof(double)*lda*lda);

    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    allocInit();

    allocCall();

    // Dune::Timer dataTrans_timer;
    // dataTrans_timer.start();
    // matricesToDevice();
    // //std::cout << "Transfer ok!" << std::endl;
    // dataTrans_timer.stop();
    // ctime_mswdatatransd += dataTrans_timer.lastElapsed();

    //umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    //umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
}

MultisegmentWellContribution::~MultisegmentWellContribution()
{
    //umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    //umfpack_di_free_numeric(&UMFPACK_Numeric);

    free(Dmatrix);

    ROCSOLVER_CALL(rocblas_destroy_handle(handle));

    freeInit();

    // HIP_CALL(hipDeviceSynchronize());
    // printf("Freeing memory at pointer %p\n", ipiv);
    // HIP_CALL(hipFree(ipiv));
    // printf("Freeing memory at pointer %p\n", d_Dmatrix_hip);
    // HIP_CALL(hipFree(d_Dmatrix_hip));
    // printf("Freeing memory at pointer %p\n", z_hip);
    // HIP_CALL(hipFree(z_hip));
    // printf("Freeing memory at pointer %p\n", d_Cvals_hip);
    // HIP_CALL(hipFree(d_Cvals_hip));
    // printf("Freeing memory at pointer %p\n", d_Bvals_hip);
    // HIP_CALL(hipFree(d_Bvals_hip));
    // printf("Freeing memory at pointer %p\n", d_Bcols_hip);
    // HIP_CALL(hipFree(d_Bcols_hip));
    // printf("Freeing memory at pointer %p\n", d_Brows_hip);
    // HIP_CALL(hipFree(d_Brows_hip));
    // printf("Freeing memory at pointer %p\n", rhs_hip);
    // HIP_CALL(hipFree(rhs_hip));
    // printf("Freeing memory at pointer %p\n", info);
    // HIP_CALL(hipFree(info));
    // printf("Freeing memory at pointer %p\n", handle);
    // ROCSOLVER_CALL(rocblas_destroy_handle(handle));

}

void checkHIPAlloc(void* ptr) {
    if (ptr == nullptr) {
        std::cerr << "HIP malloc failed." << std::endl;
        exit(1);
    }
}

void MultisegmentWellContribution::allocInit()
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
    HIP_CALL(hipMalloc(&d_rhs, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_rhs); // might be removed, is not necessary
}

void MultisegmentWellContribution::allocCall()
{
    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    checkHIPAlloc(ipiv);
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
    checkHIPAlloc(info);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
}


void MultisegmentWellContribution::matricesToDevice()
{
    Accelerator::squareCSCtoMatrix(Dmatrix, Dvals, Drows, Dcols);
    //std::cout << "  CSC to matrix ok!" << std::endl;
    //std::cout << rocM << " " << rocN << std::endl;
    HIP_CALL(hipMemcpy(d_Dmatrix, Dmatrix, rocM*rocN*sizeof(double), hipMemcpyHostToDevice));
    //std::cout << "  Dmatrix transfer ok!" << std::endl;
    HIP_CALL(hipMemcpy(d_Cvals, Cvals.data(), size(Cvals)*sizeof(double), hipMemcpyHostToDevice));
    //std::cout << "  Cvals transfer ok!" << std::endl;
    HIP_CALL(hipMemcpy(d_Bvals, Bvals.data(), size(Bvals)*sizeof(double), hipMemcpyHostToDevice));
    //std::cout << "  Bvals transfer ok!" << std::endl;
    HIP_CALL(hipMemcpy(d_Bcols, Bcols.data(), size(Bcols)*sizeof(unsigned int), hipMemcpyHostToDevice));
    //std::cout << "  Bcols transfer ok!" << std::endl;
    HIP_CALL(hipMemcpy(d_Brows, Brows.data(), size(Brows)*sizeof(unsigned int), hipMemcpyHostToDevice));
    //std::cout << "  Brows transfer ok!" << std::endl;
    HIP_CALL(hipMemcpy(d_rhs, rhs.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
    //std::cout << "  Rhs transfer ok!" << std::endl;
}

void MultisegmentWellContribution::freeInit()
{
    HIP_CALL(hipDeviceSynchronize());
    //std::cout << "  Hip sync ok!" << std::endl;
    HIP_CALL(hipFree(d_Dmatrix));
    //std::cout << "  Dmatrix ok!" << std::endl;
    HIP_CALL(hipFree(d_Cvals));
    //std::cout << "  Cvals ok!" << std::endl;
    HIP_CALL(hipFree(d_Bvals));
    //std::cout << "  Bvals ok!" << std::endl;
    HIP_CALL(hipFree(d_Bcols));
    //std::cout << "  Bcols ok!" << std::endl;
    HIP_CALL(hipFree(d_Brows));
    //std::cout << "  Brows ok!" << std::endl;
    HIP_CALL(hipFree(d_rhs));
    //std::cout << "  rhs ok!" << std::endl;

}

void MultisegmentWellContribution::freeCall()
{
    HIP_CALL(hipFree(ipiv));
    //std::cout << "  ipiv ok!" << std::endl;
    HIP_CALL(hipFree(info));
    //std::cout << "  info ok!" << std::endl;
    HIP_CALL(hipFree(d_z));
    //std::cout << "  z_hip ok!" << std::endl;
}


void MultisegmentWellContribution::solveSystem()
{
    //ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, rocM, rocN, d_Dmatrix, lda, ipiv, info));

    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix, lda, ipiv, d_z, ldb));

    HIP_CALL(hipDeviceSynchronize());
}

void MultisegmentWellContribution::blocksrmvBx(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* Rhs, double* out, unsigned int Nbr, unsigned int block_dimM, unsigned int block_dimN, const double op_sign)
{
  unsigned int blockDim = 32;
  unsigned int number_wg = std::ceil(Nbr/blockDim);
  unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
  unsigned int gridDim = num_work_groups*blockDim;
  unsigned int shared_mem_size = blockDim*sizeof(double)* block_dimM * block_dimN;

  blocksrmvBx_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size>>>(vals, cols, rows, Nbr, x, Rhs, out, block_dimM, block_dimN, op_sign);

  HIP_CALL(hipGetLastError()); // Check for errors
  HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}

/*
void MultisegmentWellContribution::blocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nb, unsigned int Nbr, unsigned int block_dimM, unsigned int block_dimN)
{
    unsigned int blockDim = 32; // Set the block size
    // unsigned int number_wg = std::ceil(Nb/blockDim);
    // unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
    // unsigned int gridDim = num_work_groups*blockDim;
    unsigned int gridDim = (Nb + blockDim - 1) / blockDim;  // Calculate grid size
    unsigned int shared_mem_size = blockDim * sizeof(double) * block_dimM * block_dimN;  // Allocate shared memory size

    blocksrmvC_z_k<<<gridDim, blockDim, shared_mem_size>>>(vals, cols, rows, Nb, Nbr, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Uncomment for synchronization if needed
}
*/

void MultisegmentWellContribution::serialBlocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nbr, int block_dimM, int block_dimN)
{
    unsigned int Nthreads = 1;
    unsigned int Nblocks = Nbr;

    dim3 block(Nthreads, 1, 1);
    dim3 grid(Nblocks, 1, 1);

    serial_blocksrmvC_z_k<<<grid, block>>>(vals, cols, rows, z, y, block_dimM, block_dimN);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Uncomment for synchronization if needed
}

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
void MultisegmentWellContribution::apply(double *d_x, double *d_y/*, double *h_x, double *h_y*/)
{
    // Dune::Timer alloc_timer;
    // alloc_timer.start();
    // allocCall();
    // //std::cout << "Alloc ok!" << std::endl;
    // alloc_timer.stop();
    // ctime_alloc += alloc_timer.lastElapsed();

    Dune::Timer dataTrans_timer;
    dataTrans_timer.start();
    matricesToDevice();
    //std::cout << "Transfer ok!" << std::endl;
    dataTrans_timer.stop();
    ctime_mswdatatransd += dataTrans_timer.lastElapsed();


    OPM_TIMEBLOCK(apply);

    Dune::Timer contribsCalc_timer;
    contribsCalc_timer.start();
    blocksrmvBx(d_Bvals, d_Bcols, d_Brows, d_x, d_rhs, d_z, size(Brows) - 1, dim_wells, dim, +1.0);
    solveSystem();
    serialBlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
    contribsCalc_timer.stop();
    ctime_welllsD += contribsCalc_timer.lastElapsed();

    // freeCall();

}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

} //namespace Opm

