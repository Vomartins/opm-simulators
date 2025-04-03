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
#include <opm/simulators/linalg/gpubridge/Reorder.hpp>

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>

#if HAVE_SUITESPARSE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif // HAVE_SUITESPARSE_UMFPACK

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
    : dim(dim_)                // size of blockvectors in vectors x and y, equal to MultisegmentWell::numEq
    , dim_wells(dim_wells_)    // size of blocks in C, B and D, equal to MultisegmentWell::numWellEq
    , M(Mb_ * dim_wells)       // number of rows, M == dim_wells*Mb
    , Mb(Mb_)                  // number of blockrows in C, D and B
    , DnumBlocks(DnumBlocks_)  // number of blocks in D
    // copy data for matrix D into vectors to prevent it going out of scope
    , Cvals(std::move(Cvalues))
    , Dvals(Dvalues, Dvalues + DnumBlocks * dim_wells * dim_wells)
    , Bvals(std::move(Bvalues))
    , Dcols(DcolPointers, DcolPointers + M + 1)
    , Bcols(std::move(BcolIndices))
    , Drows(DrowIndices, DrowIndices + DnumBlocks * dim_wells * dim_wells)
    , Brows(std::move(BrowPointers))

{
    // lsa = M
    // ldb = M
    // rocM = M
    // rocN = M
    // ipivDim = M

    z1.resize(M);
    z2.resize(M);

    Dmatrix = (double*)malloc(sizeof(double)*M*M);

    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    alloc();

    matricesToDevice();

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, M, M, d_Dmatrix, M, ipiv, info));
}

template<class Scalar>
MultisegmentWellContribution<Scalar>::~MultisegmentWellContribution()
{
    free(Dmatrix);
    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
    HIP_CALL(hipFree(d_Dmatrix));
    HIP_CALL(hipFree(d_Cvals));
    HIP_CALL(hipFree(d_Bvals));
    HIP_CALL(hipFree(d_Bcols));
    HIP_CALL(hipFree(d_Brows));
    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(info));
    HIP_CALL(hipFree(d_z));
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::alloc()
{
    HIP_CALL(hipMalloc(&d_Dmatrix, sizeof(double)*M*M));
    checkHIPAlloc(d_Dmatrix);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(Cvals)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(Bvals)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(unsigned int)*size(Bcols)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(unsigned int)*size(Brows)));
    checkHIPAlloc(d_Brows);
    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*M));
    checkHIPAlloc(ipiv);
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
    checkHIPAlloc(info);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*M*Nrhs));
    checkHIPAlloc(d_z);
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::matricesToDevice()
{
    HIP_CALL(hipMemcpy(d_Cvals, Cvals.data(), size(Cvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, Bvals.data(), size(Bvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, Bcols.data(), size(Bcols)*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, Brows.data(), size(Brows)*sizeof(unsigned int), hipMemcpyHostToDevice));

    Opm::Accelerator::squareCSCtoMatrix(Dmatrix, Dvals, Drows, Dcols);
    HIP_CALL(hipMemcpy(d_Dmatrix, Dmatrix, M*M*sizeof(double), hipMemcpyHostToDevice));
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::solveSystem()
{
    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, M, Nrhs, d_Dmatrix, M, ipiv, d_z, M));

    HIP_CALL(hipDeviceSynchronize());
}

template<class Scalar>
void MultisegmentWellContribution<Scalar>::parallelBlocksrmvB_x(double* vals,
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

template<class Scalar>
void MultisegmentWellContribution<Scalar>::parallelBlocksrmvC_z(double* vals,
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

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
template<class Scalar>
void MultisegmentWellContribution<Scalar>::apply(double *d_x, double *d_y)
{
    OPM_TIMEBLOCK(apply);

    HIP_CALL(hipMemset(d_z, 0.0, M*Nrhs*sizeof(double)));

    parallelBlocksrmvB_x(d_Bvals, d_Bcols, d_Brows, d_x, d_z, size(Brows) - 1, dim_wells, dim);
    solveSystem();
    parallelBlocksrmvC_z(d_Cvals, d_Bcols, d_Brows, d_z, d_y, size(Brows) - 1, dim, dim_wells);
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
