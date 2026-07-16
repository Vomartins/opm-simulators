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
#include <dune/common/matrixconcepts.hh>
#include <fmt/core.h>
#include <opm/simulators/linalg/system/CpuSystemBackend.hpp>
#include <opm/simulators/linalg/system/SystemPreconditionerFactory.hpp>
#include <opm/simulators/linalg/system/WellMatrixMerger.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>
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

    bool solve(Vector& x, Opm::SimulatorReportSingle* report_ptr) override
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
        // Preconditioned matrix construction for experiment 1
        // -----------------------------------------------------

        const auto currentStep = this->simulator_.episodeIndex();
        const auto totalSteps = this->simulator_.vanguard().schedule().size();
        // const auto timeStep = this->simulator_.timeStepIndex();
        const auto newtonIter = this->simulator_.problem().iterationContext().iteration();

        //std::cout << "  Total steps: " << totalSteps << std::endl;

        if (currentStep != lastReportStep_) {
            localTimeStep_ = 0;
            lastReportStep_ = currentStep;
        } else {
            if (newtonIter == 0) {
                ++localTimeStep_;
            }
        }

        std::cout << "Report step: " << currentStep << std::endl;
        //std::cout << "Time step: " << timeStep << std::endl;
        std::cout << "Newton iteration: " << newtonIter << std::endl;

        if (localTimeStep_ == 0 && newtonIter == 0) {
            const auto numOfMatrices = 6;
            const auto stepSizeMatrices = totalSteps / numOfMatrices;
            //std::cout << stepSizeMatrices << std::endl;
            auto saveMatrix = (currentStep + 1) % stepSizeMatrices + 1;
            if (currentStep == 0) saveMatrix = 1;
            if (saveMatrix == 1) {
                std::cout << "  Save matrix: " << saveMatrix << std::endl;
                std::cout << "      Report step: " << currentStep << std::endl;
                //std::cout << "      Time step: " << timeStep << std::endl;
                std::cout << "      Newton iteration: " << newtonIter << std::endl;

                size_t resBlockSize = sysRhs_[_0][1].size();
                size_t wellBlockSize = sysRhs_[_1][1].size();

                size_t N = numRes * numResDofs + numWell * numWellDofs;

                Dune::DynamicMatrix<Scalar> AM(N, N, 0.0);

                auto storeColumns = [&](const SystemVector<Scalar>& col, size_t idx)
                {
                    size_t row = 0;
                    for (size_t i = 0; i< col[_0].size(); ++i){
                        for (int c = 0; c < numResDofs; ++c){
                            AM[row++][idx] = col[_0][i][c];
                        }
                    }
                    for (size_t i = 0; i< col[_1].size(); ++i){
                        for (int c = 0; c < numWellDofs; ++c){
                            AM[row++][idx] = col[_1][i][c];
                        }
                    }
                };

                std::cout << "Number of Res Blocks:" << numRes << std::endl;
                std::cout << "  Blocks size:" << numResDofs << std::endl;
                std::cout << "Number of Well Blocks:" << numWell << std::endl;
                std::cout << "  Blocks size:" << numWellDofs << std::endl;

                std::cout << "Total DOFs: " << N << std::endl;

                size_t j_count = 0;

                for (int j = 0; j < numRes; ++j)
                {
                    for (int k = 0; k < numResDofs; ++k)
                    {
                        SystemVector<Scalar> estd = sysRhs_;
                        estd = 0.0;
                        estd[_0][j][k] = 1.0;

                        SystemVector<Scalar> Mcol = sysRhs_;
                        Mcol = 0.0;

                        sysPrecond_->apply(Mcol, estd);

                        //printSimulatorVector(Mcol, "M^{-1} column");

                        SystemVector<Scalar> AMcol = Mcol;
                        AMcol = 0.0;
                        sysMatrix_.mv(Mcol, AMcol); // AMcol = A * col = A * M^{-1} * e

                        //printSimulatorVector(AMcol, "AM^{-1} column");
                        if (j_count % 1000 == 0) std::cerr << j*numResDofs + k << " ";

                        storeColumns(AMcol, j*numResDofs + k);

                        ++j_count;
                    }
                }

                std::cout << std::endl;

                for (int j = 0; j < numWell; ++j)
                {
                    for (int k = 0; k < numWellDofs; ++k)
                    {
                        SystemVector<Scalar> estd = sysRhs_;
                        estd = 0.0;
                        estd[_1][j][k] = 1.0;

                        SystemVector<Scalar> Mcol = sysRhs_;
                        Mcol = 0.0;
                        sysPrecond_->apply(Mcol, estd);

                        SystemVector<Scalar> AMcol = Mcol;
                        AMcol = 0.0;
                        sysMatrix_.mv(Mcol, AMcol);

                        if (j_count % 1000 == 0) std::cerr << (numRes*numResDofs) + j*numWellDofs + k << " ";

                        storeColumns(AMcol, (numRes*numResDofs) + j*numWellDofs + k);

                        ++j_count;
                    }
                }

                std::cout << std::endl;

                std::cout << "j_count: " << j_count << std::endl;

                std::ofstream file(std::format("AM_inv_{}.mtx", currentStep));
                file << "%%MatrixMarket matrix array real general\n";
                file << N << " " << N << "\n";
                // MM array format is column-major
                for (size_t col = 0; col < N; ++col) {
                    for (size_t row = 0; row < N; ++row) {
                        file << AM[row][col] << "\n";
                        if (row % 1000 == 0 && col % 1000 == 0 && row == col) std::cout << row << " " << col << " /" << std::endl;
                    }
                }
                file << std::flush;
                file.close();
                std::cerr << "File written." << std::endl;
                std::cout << std::endl;
            }
        }


        // std::exit(1);




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
