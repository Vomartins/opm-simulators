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

#include <iostream>

#ifndef MULTISEGMENTWELLCONTRIBUTION_HEADER_INCLUDED
#define MULTISEGMENTWELLCONTRIBUTION_HEADER_INCLUDED

#include <vector>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if HAVE_SUITESPARSE_UMFPACK
#include<umfpack.h>
#endif
#include <dune/common/version.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

namespace Opm
{

/// This class serves to duplicate the functionality of the MultisegmentWell
/// A MultisegmentWell uses C, D and B and performs y -= (C^T * (D^-1 * (B*x)))
/// B and C are matrices, with M rows and N columns, where N is the size of the matrix. They contain blocks of MultisegmentWell::numEq by MultisegmentWell::numWellEq.
/// D is a MxM matrix, the square blocks have size MultisegmentWell::numWellEq.
/// B*x and D*B*x are a vector with M*numWellEq doubles
/// C*D*B*x is a vector with N*numEq doubles.

class MultisegmentWellContribution
{

private:
    unsigned int dim;                        // size of blockvectors in vectors x and y, equal to MultisegmentWell::numEq
    unsigned int dim_wells;                  // size of blocks in C, B and D, equal to MultisegmentWell::numWellEq
    unsigned int M;                          // number of rows, M == dim_wells*Mb
    unsigned int Mb;                         // number of blockrows in C, D and B

#if HAVE_CUDA
    cudaStream_t stream; // not actually used yet, will be when MultisegmentWellContribution are applied on GPU
#endif

    // C and B are stored in BCRS format, D is stored in CSC format (Dune::UMFPack)
    // Sparsity pattern for C is not stored, since it is the same as B
    unsigned int DnumBlocks;             // number of blocks in D
    std::vector<double> Cvals;
    std::vector<double> Dvals;
    std::vector<double> Bvals;
    std::vector<int> Dcols;              // Columnpointers, contains M+1 entries
    std::vector<unsigned int> Bcols;
    std::vector<int> Drows;              // Rowindicies, contains DnumBlocks*dim*dim_wells entries
    std::vector<unsigned int> Brows;
    std::vector<double> z1;          // z1 = B * x
    std::vector<double> z2;          // z2 = D^-1 * B * x
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;

    // RocSOLVER
    rocblas_int rocM;
    rocblas_int rocN;
    rocblas_int Nrhs = 1;
    rocblas_int lda;
    rocblas_int ldb;
    rocblas_int *info;
    rocblas_int *ipiv;
    int ipivDim;
    double *Dmatrix;
    double *d_Dmatrix;
    double *d_Cvals;
    double *d_Bvals;
    unsigned int *d_Bcols;
    unsigned int *d_Brows;
    void *d_buffer;
    rocblas_handle handle;
    rocblas_operation operation = rocblas_operation_none;
    double *d_z;
    double *d_rhs;
    std::vector<double> rhs;
    //int matrixDtransfer;

    /// Translate the columnIndex if needed
    /// Some preconditioners reorder the rows of the matrix, this means the columnIndices of the wellcontributions need to be reordered as well
    unsigned int getColIdx(unsigned int idx);

public:
    using UMFPackIndex = SuiteSparse_long;

#if HAVE_CUDA
    /// Set a cudaStream to be used
    /// \param[in] stream           the cudaStream that is used
    void setCudaStream(cudaStream_t stream);
#endif

    /// Create a new MultisegmentWellContribution
    /// Matrices C and B are passed in Blocked CSR, matrix D in CSC
    /// The variables representing C, B and D will go out of scope when MultisegmentWell::addWellContribution() ends
    /// \param[in] dim              size of blocks in blockvectors x and y, equal to MultisegmentWell::numEq
    /// \param[in] dim_wells        size of blocks of C, B and D, equal to MultisegmentWell::numWellEq
    /// \param[in] Mb               number of blockrows in C, B and D
    /// \param[in] Bvalues          nonzero values of matrix B
    /// \param[in] BcolIndices      columnindices of blocks of matrix B
    /// \param[in] BrowPointers     rowpointers of matrix B
    /// \param[in] DnumBlocks       number of blocks in D
    /// \param[in] Dvalues          nonzero values of matrix D
    /// \param[in] DcolPointers     columnpointers of matrix D
    /// \param[in] DrowIndices      rowindices of matrix D
    /// \param[in] Cvalues          nonzero values of matrix C
    MultisegmentWellContribution(unsigned int dim, unsigned int dim_wells,
                                 unsigned int Mb,
                                 std::vector<double> &Bvalues, std::vector<unsigned int> &BcolIndices, std::vector<unsigned int> &BrowPointers,
                                 unsigned int DnumBlocks, double *Dvalues, UMFPackIndex *DcolPointers,
                                 UMFPackIndex *DrowIndices, std::vector<double> &Cvalues);

    /// Destroy a MultisegmentWellContribution, and free memory
    ~MultisegmentWellContribution();

    /// Apply the MultisegmentWellContribution on CPU
    /// performs y -= (C^T * (D^-1 * (B*x))) for MultisegmentWell
    /// \param[in] h_x          vector x, must be on CPU
    /// \param[inout] h_y       vector y, must be on CPU
    void apply(double *d_x, double *d_y/*, double *h_x, double *h_y*/);

    void allocInit();

    void allocCall();

    void matricesToDevice();

    void freeInit();

    void freeCall();

    void solveSystem();

    void blocksrmvBx(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* rhs, double* out, unsigned int Nbr, unsigned int block_dimM, unsigned int block_dimN, const double op_sign);

    //void blocksrmvCtz(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* rhs, double* out, unsigned int Nb, unsigned int block_dimM, unsigned int block_dimN, const double op_sign);

    void blocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nb, unsigned int Nbr, unsigned int block_dimM, unsigned int block_dimN);

    void serialBlocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nbr,int block_dimM, int block_dimN);
};

} //namespace Opm

#endif
