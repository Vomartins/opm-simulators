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
    rocblas_status err = call;                                               \
    if (rocblas_status_success != err) {                                     \
      printf("rocSOLVER ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void saveBCRSMatrixVectors(const std::vector<double>& vecVals, const std::vector<int>& vecCols, const std::vector<int>& vecRows, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Write values vector
    size_t size1 = vecVals.size();
    outFile.write(reinterpret_cast<const char*>(&size1), sizeof(size1));
    outFile.write(reinterpret_cast<const char*>(vecVals.data()), size1 * sizeof(double));

    // Write coloumn vector
    size_t size2 = vecCols.size();
    outFile.write(reinterpret_cast<const char*>(&size2), sizeof(size2));
    outFile.write(reinterpret_cast<const char*>(vecCols.data()), size2 * sizeof(int));

    // Write row vector
    size_t size3 = vecRows.size();
    outFile.write(reinterpret_cast<const char*>(&size3), sizeof(size3));
    outFile.write(reinterpret_cast<const char*>(vecRows.data()), size3 * sizeof(int));

    outFile.close();
}

void saveBCRSMatrixVectors(const std::vector<double>& vecVals, const std::vector<unsigned int>& vecCols, const std::vector<unsigned int>& vecRows, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Write values vector
    size_t size1 = vecVals.size();
    outFile.write(reinterpret_cast<const char*>(&size1), sizeof(size1));
    outFile.write(reinterpret_cast<const char*>(vecVals.data()), size1 * sizeof(double));

    // Write coloumn vector
    size_t size2 = vecCols.size();
    std::cout << size2 << std::endl;
    outFile.write(reinterpret_cast<const char*>(&size2), sizeof(size2));
    outFile.write(reinterpret_cast<const char*>(vecCols.data()), size2 * sizeof(unsigned int));

    // Write row vector
    size_t size3 = vecRows.size();
    std::cout << size3 << std::endl;
    outFile.write(reinterpret_cast<const char*>(&size3), sizeof(size3));
    outFile.write(reinterpret_cast<const char*>(vecRows.data()), size3 * sizeof(unsigned int));

    outFile.close();
}

void saveResVector(const std::vector<double>& vecRes, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);  // Open file in binary mode
    if (!outFile) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Save vector size first to know how many elements to read back later
    size_t size = vecRes.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write the contents of the vector
    outFile.write(reinterpret_cast<const char*>(vecRes.data()), size * sizeof(double));

    outFile.close();
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
    ldb = rocM;
    ipivDim = rocM > rocN ? rocN : rocM;

    matrixDtransfer = matrixDtrans;

    // Save B, C, and D vectors
    /*
    if(well_systems == 0.0){
        char name[50];
        snprintf(name, sizeof(name), "matrix-A-%d.bin", static_cast<int>(well_counter));
        saveBCRSMatrixVectors(Dvals, Dcols, Drows, name);
        name[7] = 'B';
        saveBCRSMatrixVectors(Bvals, Bcols, Brows, name);
        name[7] = 'C';
        saveBCRSMatrixVectors(Cvals, Bcols, Brows, name);
        well_counter++;
    }
    */
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
}

void MultisegmentWellContribution::matrixDtoDevice()
{
    double* Dmatrix = Accelerator::squareCSCtoMatrix(Dvals, Drows, Dcols);

    HIP_CALL(hipMemcpy(d_Dmatrix_hip, Dmatrix, rocM*rocN*sizeof(double), hipMemcpyHostToDevice));
}

void MultisegmentWellContribution::freeRocSOLVER()
{
    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(d_Dmatrix_hip));
    HIP_CALL(hipFree(z_hip));
    HIP_CALL(hipFree(info));
    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
}

void MultisegmentWellContribution::solveSystem()
{
    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, rocM, rocN, d_Dmatrix_hip, lda, ipiv, info));

    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, rocN, Nrhs, d_Dmatrix_hip, lda, ipiv, z_hip, ldb));
}

// Apply the MultisegmentWellContribution, similar to MultisegmentWell::apply()
// h_x and h_y reside on host
// y -= (C^T * (D^-1 * (B * x)))
void MultisegmentWellContribution::apply(double *h_x, double *h_y)
{

    OPM_TIMEBLOCK(apply);
    // reset z1 and z2
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

    Dune::Timer alloc_timer;
    alloc_timer.start();
    hipAlloc();
    alloc_timer.stop();
    ctime_alloc += alloc_timer.lastElapsed();

    Dune::Timer dataTransWell_timer;
    dataTransWell_timer.start();
    HIP_CALL(hipMemcpy(z_hip, z1.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
    dataTransWell_timer.stop();
    ctime_rocsoldatatrans += dataTransWell_timer.lastElapsed();

    if(matrixDtransfer == 0){
        Dune::Timer dataTransD_timer;
        dataTransD_timer.start();
        matrixDtoDevice();
        dataTransD_timer.stop();
        ctime_mswdatatransd += dataTransD_timer.lastElapsed();
        //matrixDtransfer++;
    }

    Dune::Timer linearSysD_timer;
    linearSysD_timer.start();
    solveSystem();
    linearSysD_timer.stop();
    ctime_welllsD += linearSysD_timer.lastElapsed();

    dataTransWell_timer.start();
    HIP_CALL(hipMemcpy(z2.data(), z_hip, rocM*sizeof(double),hipMemcpyDeviceToHost));
    dataTransWell_timer.stop();
    ctime_rocsoldatatrans += dataTransWell_timer.lastElapsed();

    freeRocSOLVER();

    /*
    char name[50];
    if(well_systems == 0.0){
        snprintf(name, sizeof(name), "vector-Res-%d.bin", static_cast<int>(vec_counter));
        size_t size = (*std::max_element(Bcols.begin(),Bcols.end())+1)*dim;
        std::cout << size << std::endl;
        std::vector<double> vech_x(h_x,h_x+size);
        saveResVector(vech_x, name);
        std::cout << Mb << std::endl;
        std::cout << dim << std::endl;
        std::cout << dim_wells << std::endl;
    }
    */
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
    if(well_systems == 0.0){
        name[7] = 'S';
        name[8] = 'o';
        name[9] = 'l';
        saveResVector(z2, name);
        vec_counter++;
    }
    */
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
}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

} //namespace Opm

