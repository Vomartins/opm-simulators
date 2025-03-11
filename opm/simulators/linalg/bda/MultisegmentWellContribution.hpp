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
#include <rocsparse/rocsparse.h>

#include <opm/simulators/timestepping/SimulatorReport.hpp>

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
public:
    unsigned int Mb;

private:
    unsigned int dim;                        // size of blockvectors in vectors x and y, equal to MultisegmentWell::numEq
    unsigned int dim_wells;                  // size of blocks in C, B and D, equal to MultisegmentWell::numWellEq
    unsigned int M;                          // number of rows, M == dim_wells*Mb
    //unsigned int Mb;                         // number of blockrows in C, D and B

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
    std::vector<int> Drows;              // Rowindicies, contains DnumBlocks*dim_wells*dim_wells entries
    std::vector<unsigned int> Brows;
    std::vector<double> z1;          // z1 = B * x
    std::vector<double> z2;          // z2 = D^-1 * B * x

    // RocSPARSE
    // Auxiliary vectors to convert D to CSR format
    std::vector<double> Dvals_;
    std::vector<int> Dcols_;
    std::vector<int> Drows_;

    double one  = 1.0;
    rocsparse_int rocM;
    rocsparse_int rocN;
    rocsparse_int Nrhs = 1;
    rocsparse_int lda;
    rocsparse_int ldb;
    //rocsparse_mat_info ilu_info;
    std::vector<rocsparse_mat_info> ilu_info;
    std::vector<rocsparse_mat_info> L_info;
    std::vector<rocsparse_mat_info> U_info;
    rocsparse_mat_descr descr_D, descr_L, descr_U;
    std::size_t d_bufferSize_D, d_bufferSize_L, d_bufferSize_U, d_bufferSize;
    void *d_buffer_D, *d_buffer_L, *d_buffer_U;
    rocsparse_handle handle;
    rocsparse_operation operation = rocsparse_operation_none;
    rocsparse_int nnzs;

    // Device arrays
    double *d_Dvals;
    rocsparse_int *d_Dcols;
    rocsparse_int *d_Drows;
    double *d_Cvals;
    double *d_Bvals;
    unsigned int *d_Bcols;
    unsigned int *d_Brows;
    double *d_z;
    double *d_z_aux;
    double *d_rhs;
    double *d_x_elem;         // Auxiliary array to multiply Bw*xr in a contiguous memory access

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

    /// Apply the MultisegmentWellContribution on GPU
    /// performs y -= (C^T * (D^-1 * (B*x))) for MultisegmentWell
    /// \param[in] d_x          vector x, must be on GPU
    /// \param[inout] d_y       vector y, must be on GPU
    void apply(double *d_x, double *d_y);

    void allocInit();

    void allocCall();

    void matricesToDevice();

    void freeInit();

    void freeCall();

    void squareCSCtoCSR(std::vector<double> vals, std::vector<int> rows, std::vector<int> cols, std::vector<double>& vals_, std::vector<int>& rows_, std::vector<int>& cols_);

    void analyseMatrix();

    void solveSystem();

    void blocksrmvBx(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* out, unsigned int Nbr, unsigned int block_dimM, unsigned int block_dimN, const double op_sign);

    void serialBlocksrmvB_x(double* vals,
                            unsigned int* cols,
                            unsigned int* rows,
                            double* x,
                            double* y,
                            unsigned int Nbr,
                            int block_dimM,
                            int block_dimN);

    void parallelBlocksrmvB_x(double* vals,
                              unsigned int* cols,
                              unsigned int* rows,
                              double* x,
                              double* y,
                              unsigned int Nbr,
                              int block_dimM,
                              int block_dimN);

    void parallelV2BlocksrmvB_x(double* vals,
                                unsigned int* cols,
                                unsigned int* rows,
                                double* x,
                                double* y,
                                unsigned int Nbr,
                                int block_dimM,
                                int block_dimN);

    void parallelV3BlocksrmvB_x(double* vals,
                              unsigned int* cols,
                              unsigned int* rows,
                              double* x,
                              double* x_elem,
                              double* y,
                              unsigned int Nbr,
                              int block_dimM,
                              int block_dimN);

    void serialBlocksrmvC_z(double* vals,
                            unsigned int* cols,
                            unsigned int* rows,
                            double* z,
                            double* y,
                            unsigned int Nbr,
                            int block_dimM,
                            int block_dimN);

    void parallelBlocksrmvC_z(double* vals,
                              unsigned int* cols,
                              unsigned int* rows,
                              double* z,
                              double* y,
                              unsigned int Nbr,
                              int block_dimM,
                              int block_dimN);

    void parallelV2BlocksrmvC_z(double* vals,
                                unsigned int* cols,
                                unsigned int* rows,
                                double* z,
                                double* y,
                                unsigned int Nbr,
                                int block_dimM,
                                int block_dimN);

};

} //namespace Opm

#endif
