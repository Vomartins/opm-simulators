/*
  Copyright Equinor ASA 2026

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
#ifndef OPM_ISTLSOLVERSYSTEM_HEADER_INCLUDED
#define OPM_ISTLSOLVERSYSTEM_HEADER_INCLUDED

#include "SystemTypes.hpp"
#include <cstddef>
#include <dune/common/matrixconcepts.hh>
#include <fmt/core.h>
#include <opm/simulators/linalg/system/CpuSystemBackend.hpp>
#include <opm/simulators/linalg/system/SystemPreconditionerFactory.hpp>
#include <opm/simulators/linalg/system/WellMatrixMerger.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace Opm
{

template <class TypeTag>
class ISTLSolverSystem : public ISTLSolver<TypeTag>
{
protected:
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Vector = GetPropType<TypeTag, Properties::GlobalEqVector>;
    using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;
    using Matrix = typename SparseMatrixAdapter::IstlMatrix;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;

    mutable int localTimeStep_ = 0;
    mutable int lastReportStep_ = -1;
    mutable int lastSavedStep_ = -1;

    // Compile-time validation: SystemPreconditionerFactory and related types
    // are hardcoded for standard 3-phase blackoil (3 reservoir equations, 4 well equations).
    // See SystemTypes.hpp for details.
    static_assert(Indices::numEq == 3,
                  "ISTLSolverSystem (with system_cpr preconditioner) only supports "
                  "3-equation blackoil models. This model has different equation count.");

    constexpr static std::size_t pressureIndex
        = Indices::pressureSwitchIdx;

    enum { enablePolymerMolarWeight = getPropValue<TypeTag, Properties::EnablePolymerMW>() };
    constexpr static bool isIncompatibleWithCprw = enablePolymerMolarWeight;

#if HAVE_MPI
    using CommunicationType = Dune::OwnerOverlapCopyCommunication<int, int>;
#else
    using CommunicationType = Dune::Communication<int>;
#endif
    using Parent = ISTLSolver<TypeTag>;

    static constexpr auto _0 = Dune::Indices::_0;
    static constexpr auto _1 = Dune::Indices::_1;

public:
    ISTLSolverSystem(const Simulator& simulator,
                     const FlowLinearSolverParameters& parameters,
                     bool forceSerial = false)
        : Parent(simulator, parameters, forceSerial)
    {
    }

    explicit ISTLSolverSystem(const Simulator& simulator)
        : Parent(simulator)
    {
    }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        OPM_TIMEBLOCK(istlSolverPrepare);
        this->initPrepare(M.istlMatrix(), b);
        prepareSystemSolver();
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        OPM_TIMEBLOCK(istlSolverPrepare);
        this->initPrepare(M, b);
        prepareSystemSolver();
    }

    void printSimulatorVector(const SystemVector<Scalar>& vec,const std::string& name)
    {
        std::cout << name << ": " << std::endl;
        for (std::size_t i = 0; i < 10; ++i) {
            const auto& block = vec[_0][i];
            std::cout << "res[" << i << "] = ";
            for (int c = 0; c < block.size(); ++c){
                std::cout << block[c] << " ";
            }
            std::cout << std::endl;
        }
        for (std::size_t i = 0; i < 10; ++i) {
            const auto& block = vec[_1][i];
            std::cout << "well[" << i << "] = ";
            for (int c = 0; c < block.size(); ++c){
                std::cout << block[c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    bool solve(Vector& x, Opm::SimulatorReportSingle* report_ptr = nullptr) override
    {
        OPM_TIMEBLOCK(istlSolverSolve);
        ++this->solveCount_;

        using SysPrecSeq = SystemPreconditioner<CpuSystemBackend<Scalar, SeqResOperatorT<Scalar>>>;
        if (auto* p = dynamic_cast<SysPrecSeq*>(sysPrecond_)) {
            p->setSimulatorReportPointer(report_ptr);
        }

        const std::size_t numRes = Parent::matrix_->N();
        const std::size_t numWell = cachedWellStructure_.totalWellBlocks;
        sysX_[_0].resize(numRes);
        sysX_[_0] = 0.0;
        sysX_[_1].resize(numWell);
        sysX_[_1] = 0.0;

        sysRhs_[_0].resize(numRes);
        sysRhs_[_0] = *Parent::rhs_;
        sysRhs_[_1].resize(numWell);
        sysRhs_[_1] = 0.0;

        Dune::InverseOperatorResult result;
        sysSolver_->apply(sysX_, sysRhs_, result);
        this->iterations_ = result.iterations;

        x = sysX_[_0];

        // -----------------------------------------------------
        // Matrix extraction for experiment 1
        // Saves: T = S*M^{-1}, S (system matrix),
        //        M1^{-1} (res CPR), M2^{-1} (well solver),
        //        M3^{-1} (res smoother)
        // -----------------------------------------------------

//         const auto currentStep = this->simulator_.episodeIndex();
//         const auto totalSteps = this->simulator_.vanguard().schedule().size();
//         const auto newtonIter = this->simulator_.problem().iterationContext().iteration();
//
//         if (currentStep != lastReportStep_) {
//             localTimeStep_ = 0;
//             lastReportStep_ = currentStep;
//         } else {
//             if (newtonIter == 0) {
//                 ++localTimeStep_;
//             }
//         }
//
//         std::cout << "Report step: " << currentStep << std::endl;
//         std::cout << "Newton iteration: " << newtonIter << std::endl;
//
//         if (localTimeStep_ == 0 && newtonIter == 0 && currentStep != lastSavedStep_) {
//             const auto numOfMatrices = 6;
//             const auto stepSizeMatrices = totalSteps / numOfMatrices;
//             auto saveMatrix = (currentStep + 1) % stepSizeMatrices + 1;
//             if (currentStep == 0) saveMatrix = 1;
//             if (saveMatrix == 1) {
//                 std::cout << "  Save matrix: " << saveMatrix << std::endl;
//                 std::cout << "      Report step: " << currentStep << std::endl;
//                 std::cout << "      Newton iteration: " << newtonIter << std::endl;
//
//                 const size_t nResDofs = numRes * numResDofs;
//                 const size_t nWellDofs = numWell * numWellDofs;
//                 const size_t N = nResDofs + nWellDofs;
//
//                 // Get concrete preconditioner for individual stage access
//                 auto* precond = dynamic_cast<SysPrecSeq*>(sysPrecond_);
//
//                 // Dense matrices: full preconditioned system and individual stages
//                 Dune::DynamicMatrix<Scalar> AM(N, N, 0.0);
//                 Dune::DynamicMatrix<Scalar> M1(nResDofs, nResDofs, 0.0);
//                 Dune::DynamicMatrix<Scalar> M3(nResDofs, nResDofs, 0.0);
//                 Dune::DynamicMatrix<Scalar> M2(nWellDofs, nWellDofs, 0.0);
//
//                 auto storeSysColumn = [&](const SystemVector<Scalar>& col,
//                                           Dune::DynamicMatrix<Scalar>& mat, size_t idx)
//                 {
//                     size_t row = 0;
//                     for (size_t i = 0; i < col[_0].size(); ++i)
//                         for (int c = 0; c < numResDofs; ++c)
//                             mat[row++][idx] = col[_0][i][c];
//                     for (size_t i = 0; i < col[_1].size(); ++i)
//                         for (int c = 0; c < numWellDofs; ++c)
//                             mat[row++][idx] = col[_1][i][c];
//                 };
//
//                 auto storeResColumn = [](const ResVector<Scalar>& col,
//                                          Dune::DynamicMatrix<Scalar>& mat, size_t idx)
//                 {
//                     size_t row = 0;
//                     for (size_t i = 0; i < col.size(); ++i)
//                         for (int c = 0; c < numResDofs; ++c)
//                             mat[row++][idx] = col[i][c];
//                 };
//
//                 auto storeWellColumn = [](const WellVector<Scalar>& col,
//                                           Dune::DynamicMatrix<Scalar>& mat, size_t idx)
//                 {
//                     size_t row = 0;
//                     for (size_t i = 0; i < col.size(); ++i)
//                         for (int c = 0; c < numWellDofs; ++c)
//                             mat[row++][idx] = col[i][c];
//                 };
//
//                 // Pre-allocate work vectors for stage probing
//                 ResVector<Scalar> resE(numRes), resSol(numRes), resTmp(numRes);
//                 WellVector<Scalar> wellE(numWell), wellSol(numWell);
//
//                 std::cout << "Number of Res Blocks:" << numRes << std::endl;
//                 std::cout << "  Blocks size:" << numResDofs << std::endl;
//                 std::cout << "Number of Well Blocks:" << numWell << std::endl;
//                 std::cout << "  Blocks size:" << numWellDofs << std::endl;
//                 std::cout << "Total DOFs: " << N << std::endl;
//
//                 size_t j_count = 0;
//
//                 size_t probeFailures = 0;
//
//                 // Reservoir basis vectors: probe T, M1 (CPR), M3 (smoother)
//                 for (size_t j = 0; j < numRes; ++j)
//                 {
//                     for (int k = 0; k < numResDofs; ++k)
//                     {
//                         const size_t colIdx = j * numResDofs + k;
//                         try {
//                             // Full preconditioner: T = S * M^{-1}
//                             SystemVector<Scalar> estd = sysRhs_;
//                             estd = 0.0;
//                             estd[_0][j][k] = 1.0;
//                             SystemVector<Scalar> Mcol = sysRhs_;
//                             Mcol = 0.0;
//                             sysPrecond_->apply(Mcol, estd);
//                             SystemVector<Scalar> AMcol = Mcol;
//                             AMcol = 0.0;
//                             sysMatrix_.mv(Mcol, AMcol);
//                             storeSysColumn(AMcol, AM, colIdx);
//
//                             // Individual stage probing
//                             if (precond) {
//                                 // Stage 1: reservoir CPR solve
//                                 resE = 0.0;
//                                 resE[j][k] = 1.0;
//                                 resSol = 0.0;
//                                 resTmp = resE;
//                                 Dune::InverseOperatorResult res_result;
//                                 precond->resSolver().apply(resSol, resTmp, res_result);
//                                 storeResColumn(resSol, M1, colIdx);
//
//                                 // Stage 2b: reservoir smoother
//                                 resSol = 0.0;
//                                 resTmp = resE;
//                                 precond->resSmoother().apply(resSol, resTmp, res_result);
//                                 storeResColumn(resSol, M3, colIdx);
//                             }
//                         } catch (const std::exception& e) {
//                             if (probeFailures == 0)
//                                 std::cerr << "Probe failure at res col " << colIdx << ": " << e.what() << std::endl;
//                             ++probeFailures;
//                         }
//
//                         if (j_count % 1000 == 0) std::cerr << colIdx << " ";
//                         ++j_count;
//                     }
//                 }
//
//                 std::cout << std::endl;
//
//                 // Well basis vectors: probe T and M2 (well solver)
//                 for (size_t j = 0; j < numWell; ++j)
//                 {
//                     for (int k = 0; k < numWellDofs; ++k)
//                     {
//                         const size_t sysColIdx = nResDofs + j * numWellDofs + k;
//                         const size_t wellColIdx = j * numWellDofs + k;
//                         try {
//                             // Full preconditioner: T = S * M^{-1}
//                             SystemVector<Scalar> estd = sysRhs_;
//                             estd = 0.0;
//                             estd[_1][j][k] = 1.0;
//                             SystemVector<Scalar> Mcol = sysRhs_;
//                             Mcol = 0.0;
//                             sysPrecond_->apply(Mcol, estd);
//                             SystemVector<Scalar> AMcol = Mcol;
//                             AMcol = 0.0;
//                             sysMatrix_.mv(Mcol, AMcol);
//                             storeSysColumn(AMcol, AM, sysColIdx);
//
//                             // Stage 2a / Stage 3: well solver
//                             if (precond) {
//                                 wellE = 0.0;
//                                 wellE[j][k] = 1.0;
//                                 wellSol = 0.0;
//                                 precond->applyWellSolve(wellSol, wellE);
//                                 storeWellColumn(wellSol, M2, wellColIdx);
//                             }
//                         } catch (const std::exception& e) {
//                             if (probeFailures == 0)
//                                 std::cerr << "Probe failure at well col " << wellColIdx << ": " << e.what() << std::endl;
//                             ++probeFailures;
//                         }
//
//                         if (j_count % 1000 == 0) std::cerr << sysColIdx << " ";
//                         ++j_count;
//                     }
//                 }
//
//                 std::cout << std::endl;
//                 std::cout << "j_count: " << j_count << std::endl;
//                 if (probeFailures > 0)
//                     std::cerr << "Total probe failures: " << probeFailures << " / " << j_count
//                               << " (failed columns stored as zero)" << std::endl;
//
//                 // Save dense matrix in MatrixMarket array format
//                 auto saveDenseMatrix = [](const Dune::DynamicMatrix<Scalar>& mat,
//                                           size_t rows, size_t cols,
//                                           const std::string& filename)
//                 {
//                     std::ofstream file(filename);
//                     file << std::setprecision(17);
//                     file << "%%MatrixMarket matrix array real general\n";
//                     file << rows << " " << cols << "\n";
//                     for (size_t c = 0; c < cols; ++c)
//                         for (size_t r = 0; r < rows; ++r)
//                             file << mat[r][c] << "\n";
//                     file << std::flush;
//                     file.close();
//                     std::cerr << filename << " written." << std::endl;
//                 };
//
//                 saveDenseMatrix(AM, N, N, std::format("AM_inv_{}.mtx", currentStep));
//                 if (precond) {
//                     saveDenseMatrix(M1, nResDofs, nResDofs, std::format("res_cpr_{}.mtx", currentStep));
//                     saveDenseMatrix(M3, nResDofs, nResDofs, std::format("res_smoother_{}.mtx", currentStep));
//                     saveDenseMatrix(M2, nWellDofs, nWellDofs, std::format("well_solver_{}.mtx", currentStep));
//                 }
//
//                 // Save system matrix (sparse, coordinate format)
//                 // Layout: [ A  C ]   A: numRes*numRes (3x3 blocks)
//                 //         [ B  D ]   B: numWell*numRes (4x3), C: numRes*numWell (3x4), D: numWell*numWell (4x4)
//                 {
//                     const size_t rowOffsetWell = nResDofs;
//                     const size_t colOffsetWell = nResDofs;
//
//                     auto countBCRS = [](const auto& mat, int bRows, int bCols) -> size_t
//                     {
//                         size_t count = 0;
//                         for (auto rowIt = mat.begin(); rowIt != mat.end(); ++rowIt)
//                             for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
//                                 count += bRows * bCols;
//                         return count;
//                     };
//
//                     auto writeBCRS = [](std::ofstream& out, const auto& mat,
//                                         size_t rowOff, size_t colOff,
//                                         int bRows, int bCols)
//                     {
//                         for (auto rowIt = mat.begin(); rowIt != mat.end(); ++rowIt) {
//                             for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt) {
//                                 const auto& block = *colIt;
//                                 for (int lr = 0; lr < bRows; ++lr) {
//                                     for (int lc = 0; lc < bCols; ++lc) {
//                                         const size_t globalRow = rowOff + rowIt.index() * bRows + lr + 1;
//                                         const size_t globalCol = colOff + colIt.index() * bCols + lc + 1;
//                                         out << globalRow << " " << globalCol << " " << block[lr][lc] << "\n";
//                                     }
//                                 }
//                             }
//                         }
//                     };
//
//                     size_t nnzTotal = 0;
//                     nnzTotal += countBCRS(*sysMatrix_.A, numResDofs, numResDofs);
//                     nnzTotal += countBCRS(*sysMatrix_.C, numResDofs, numWellDofs);
//                     nnzTotal += countBCRS(*sysMatrix_.B, numWellDofs, numResDofs);
//                     nnzTotal += countBCRS(*sysMatrix_.D, numWellDofs, numWellDofs);
//
//                     std::ofstream sysFile(std::format("system_matrix_{}.mtx", currentStep));
//                     sysFile << std::setprecision(17);
//                     sysFile << "%%MatrixMarket matrix coordinate real general\n";
//                     sysFile << N << " " << N << " " << nnzTotal << "\n";
//
//                     writeBCRS(sysFile, *sysMatrix_.A, 0,            0,             numResDofs,  numResDofs);
//                     writeBCRS(sysFile, *sysMatrix_.C, 0,            colOffsetWell, numResDofs,  numWellDofs);
//                     writeBCRS(sysFile, *sysMatrix_.B, rowOffsetWell, 0,             numWellDofs, numResDofs);
//                     writeBCRS(sysFile, *sysMatrix_.D, rowOffsetWell, colOffsetWell, numWellDofs, numWellDofs);
//
//                     sysFile << std::flush;
//                     sysFile.close();
//                     std::cerr << "System matrix file written (" << nnzTotal << " non-zeros)." << std::endl;
//                 }
//                 lastSavedStep_ = currentStep;
//             }
//         }

        return this->checkConvergence(result);
    }

private:
    bool sysInitialized_ = false;
    WellMatrixStructure cachedWellStructure_;

    // Current per-well B/C/D blocks for the explicit 2x2 system matrix.
    std::vector<WRMatrix<Scalar>> wellBMatrices_;
    std::vector<RWMatrix<Scalar>> wellCMatrices_;
    std::vector<WWMatrix<Scalar>> wellDMatrices_;
    Opm::SparseTable<int> wellCells_;

    // Owned storage for merged well matrices; SystemMatrix points into these.
    WRMatrix<Scalar> mergedB_;
    RWMatrix<Scalar> mergedC_;
    WWMatrix<Scalar> mergedD_;

    SystemMatrix<Scalar> sysMatrix_;
    SystemVector<Scalar> sysX_;
    SystemVector<Scalar> sysRhs_;

    // Serial solver components
    std::unique_ptr<SystemSeqOp<Scalar>> sysOp_;
    std::unique_ptr<Dune::FlexibleSolver<SystemSeqOp<Scalar>>> sysFlexSolverSeq_;

    // Parallel solver components
#if HAVE_MPI
    using WellComm = Dune::JacComm;
    std::unique_ptr<WellComm> wellComm_;
    std::unique_ptr<SystemComm> systemComm_;
    std::unique_ptr<SystemParOp<Scalar>> sysOpPar_;
    std::unique_ptr<Dune::FlexibleSolver<SystemParOp<Scalar>>> sysFlexSolverPar_;
#endif

    using SysSolverType = Dune::InverseOperator<SystemVector<Scalar>, SystemVector<Scalar>>;
    using SysPrecondType = Dune::PreconditionerWithUpdate<SystemVector<Scalar>, SystemVector<Scalar>>;
    using SeqSysPrecondType = SystemPreconditioner<CpuSystemBackend<Scalar, SeqResOperatorT<Scalar>>>;
#if HAVE_MPI
    using ParSysPrecondType = SystemPreconditioner<CpuSystemBackend<Scalar, ParResOperatorT<Scalar>, ParResComm>>;
#endif
    SysSolverType* sysSolver_ = nullptr;
    SysPrecondType* sysPrecond_ = nullptr;

    void prepareSystemSolver()
    {
        OPM_TIMEBLOCK(flexibleSolverPrepare);

        wellBMatrices_.clear();
        wellCMatrices_.clear();
        wellDMatrices_.clear();
        wellCells_.clear();

        this->simulator_.problem().wellModel().addBCDMatrix(
            wellBMatrices_, wellCMatrices_, wellDMatrices_, wellCells_);

        const Opm::WellMatrixMerger<Scalar> merger(
            Parent::matrix_->N(), wellBMatrices_, wellCMatrices_, wellDMatrices_, wellCells_);

        const bool localStructureChanged = !sysInitialized_
            || !merger.hasSameStructure(cachedWellStructure_);

        // All ranks must agree on whether to take the structure-change path,
        // because the distributed solver create and update paths use different
        // MPI-collective sequences.
#if HAVE_MPI
        const bool globalStructureChanged = this->comm_->communicator().max(
            static_cast<int>(localStructureChanged)) > 0;
#else
        const bool globalStructureChanged = localStructureChanged;
#endif
        const bool needStructureRefresh = !sysInitialized_ || globalStructureChanged;

        const auto& prm = this->prm_[this->activeSolverNum_];

        if (needStructureRefresh) {
            OPM_TIMEBLOCK(flexibleSolverCreate);
            merger.buildMatrices(mergedB_, mergedC_, mergedD_);
            sysMatrix_.A = Parent::matrix_;
            sysMatrix_.B = &mergedB_;
            sysMatrix_.C = &mergedC_;
            sysMatrix_.D = &mergedD_;
            cachedWellStructure_ = merger.buildStructure();

            refreshSystemSolverForChangedWellStructure(prm);
            sysInitialized_ = true;
        } else {
            OPM_TIMEBLOCK(flexibleSolverUpdate);
            // Pattern unchanged: write fresh values into the existing merged
            // matrices without any (de)allocation.
            merger.updateValues(mergedB_, mergedC_, mergedD_);

            // Refresh A pointer in case the reservoir matrix was reallocated.
            sysMatrix_.A = Parent::matrix_;
            sysMatrix_.B = &mergedB_;
            sysMatrix_.C = &mergedC_;
            sysMatrix_.D = &mergedD_;
            sysPrecond_->update();
        }
    }

    void refreshSystemSolverForChangedWellStructure(const Opm::PropertyTree& prm)
    {
        // When the well structure changes, rebuild the system solver from scratch.
        // SystemPreconditioner does not support incremental well-structure updates.
        createSystemSolver(prm);
    }

    void createSystemSolver(const Opm::PropertyTree& prm)
    {
        // Derive weights from the reservoir sub-block config (which uses CPR internally)
        auto resSolverPrm = prm.get_child("preconditioner.reservoir_solver");
        std::function<ResVector<Scalar>()> resWeightCalc
            = this->getWeightsCalculator(resSolverPrm, this->getMatrix(), pressureIndex);

        std::function<SystemVector<Scalar>()> sysWeightCalc;
        if (resWeightCalc) {
            sysWeightCalc = [resWeightCalc]() {
                SystemVector<Scalar> w;
                w[_0] = resWeightCalc();
                return w;
            };
        }

#if HAVE_MPI
        const bool is_parallel = this->comm_->communicator().size() > 1;
        if (is_parallel) {
            wellComm_ = std::make_unique<WellComm>();
            systemComm_ = std::make_unique<SystemComm>(*(this->comm_), *wellComm_);

            sysOpPar_ = std::make_unique<SystemParOp<Scalar>>(sysMatrix_, *systemComm_);

            sysFlexSolverPar_ = std::make_unique<Dune::FlexibleSolver<SystemParOp<Scalar>>>(
                *sysOpPar_, *systemComm_, prm, sysWeightCalc, pressureIndex);

            sysSolver_ = sysFlexSolverPar_.get();
            sysPrecond_ = &sysFlexSolverPar_->preconditioner();
        }
        else
#endif
        {
            sysOp_ = std::make_unique<SystemSeqOp<Scalar>>(sysMatrix_);

            sysFlexSolverSeq_ = std::make_unique<Dune::FlexibleSolver<SystemSeqOp<Scalar>>>(
                *sysOp_, prm, sysWeightCalc, pressureIndex);

            sysSolver_ = sysFlexSolverSeq_.get();
            sysPrecond_ = &sysFlexSolverSeq_->preconditioner();
        }
    }
};

} // namespace Opm

#endif // OPM_ISTLSOLVERSYSTEM_HEADER_INCLUDED
