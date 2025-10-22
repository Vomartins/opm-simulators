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
// #include <dune/common/timer.hh>

extern double msw_alloc;
extern double msw_dataTrans;
extern double msw_LU;
extern double msw_lsD;
extern double msw_Bx;
extern double msw_Cz;

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

    z1.resize(Mb * dim_wells);
    z2.resize(Mb * dim_wells);

    Dune::Timer LU_timer;
    LU_timer.start();
    umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
    LU_timer.stop();
    msw_LU += LU_timer.lastElapsed();
}

MultisegmentWellContribution::~MultisegmentWellContribution()
{
    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);
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

    // saveVector(Bvals, "Bvals.txt");
    // saveVector(Cvals, "Cvals.txt");
    // saveVector(Bcols, "Bcols.txt");
    // saveVector(Brows, "Brows.txt");
    // saveVector(Dvals, "Dvals.txt");
    // saveVector(Dcols, "Dcols.txt");
    // saveVector(Drows, "Drows.txt");
    // exit(1);

    // z1 = B * x
    Dune::Timer Bx_timer;
    Bx_timer.start();
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
    Bx_timer.stop();
    msw_Bx += Bx_timer.lastElapsed();

    // z2 = D^-1 * (B * x)
    // umfpack
    Dune::Timer lsD_timer;
    lsD_timer.start();
    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);
    lsD_timer.stop();
    msw_lsD += lsD_timer.lastElapsed();

    // y -= (C^T * z2)
    // y -= (C^T * (D^-1 * (B * x)))
    Dune::Timer Cz_timer;
    Cz_timer.start();
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
    Cz_timer.stop();
    msw_Cz += Cz_timer.lastElapsed();
}

#if HAVE_CUDA
void MultisegmentWellContribution::setCudaStream(cudaStream_t stream_)
{
    stream = stream_;
}
#endif

} //namespace Opm
