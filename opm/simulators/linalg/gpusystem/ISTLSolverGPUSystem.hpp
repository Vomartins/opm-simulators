/*
  Copyright 2025 Equinor ASA

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

#pragma once

// -------------------------------------------------------------------------
// Includes — GPU system types and preconditioner
// -------------------------------------------------------------------------
#include <opm/simulators/linalg/gpusystem/GpuSystemBackend.hpp>
#include <opm/simulators/linalg/gpusystem/GpuSystemPreconditionerFactory.hpp>
#include <opm/simulators/linalg/gpusystem/GpuSystemTypes.hpp>
#include <opm/simulators/linalg/system/SystemPreconditioner.hpp>

// CPU well matrix merging
#include <opm/simulators/linalg/system/SystemTypes.hpp>
#include <opm/simulators/linalg/system/WellMatrixMerger.hpp>

// OPM linalg
#include <opm/simulators/linalg/AbstractISTLSolver.hpp>
#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>

// GPU primitives
#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuVector.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#endif

// opm-models / grid
#include <opm/grid/utility/ElementChunks.hpp>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/simulators/linalg/ExtractParallelGridInformationToISTL.hpp>
#include <opm/simulators/linalg/findOverlapRowsAndColumns.hpp>
#include <opm/simulators/timestepping/SimulatorReport.hpp>

// Dune
#include <dune/common/parallel/communication.hh>
#include <dune/istl/operators.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Opm::gpusystem
{

// --------------------------------------------------------------------------
// ISTLSolverGPUSystem
//
// GPU analogue of ISTLSolverSystem.  Runs the 3-stage SystemPreconditioner<GpuSystemBackend>
// inside a Dune::FlexibleSolver over the coupled (res+well) GPU system.
//
// Derive from AbstractISTLSolver<SparseMatrixAdapter, Vector> directly; all GPU infrastructure
// (comms, prm, weights) is owned here to avoid the private-member barrier
// of ISTLSolverGPUISTL.
//
// Constraints:
//   - --matrix-add-well-contributions=false   (well blocks handled separately)
//   - Sequential only (MPI parallel not yet supported for the coupled system)
// --------------------------------------------------------------------------
template<class TypeTag>
class ISTLSolverGPUSystem : public AbstractISTLSolver<GetPropType<TypeTag, Properties::SparseMatrixAdapter>,
                                                      GetPropType<TypeTag, Properties::GlobalEqVector>>
{
    // ------------------------------------------------------------------
    // Type aliases
    // ------------------------------------------------------------------
    using Simulator     = GetPropType<TypeTag, Properties::Simulator>;
    using SparseMatrixAdapter
                        = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;
    using Vector        = GetPropType<TypeTag, Properties::GlobalEqVector>;
    using Parent        = AbstractISTLSolver<SparseMatrixAdapter, Vector>;
    using Matrix        = typename SparseMatrixAdapter::IstlMatrix;
    using Scalar        = GetPropType<TypeTag, Properties::Scalar>;
    using Indices       = GetPropType<TypeTag, Properties::Indices>;
    using GridView      = GetPropType<TypeTag, Properties::GridView>;
    using ThreadManager = GetPropType<TypeTag, Properties::ThreadManager>;
    using ElementContext= GetPropType<TypeTag, Properties::ElementContext>;
    using ElementChunksType
                        = Opm::ElementChunks<GridView, Dune::Partitions::All>;

#if HAVE_MPI
    using CommunicationType = Dune::OwnerOverlapCopyCommunication<int, int>;
#else
    using CommunicationType = Dune::Communication<int>;
#endif

    using GpuSparseMatrix = gpuistl::GpuSparseMatrixWrapper<Scalar>;
    using GpuVec          = gpuistl::GpuVector<Scalar>;
    using GpuVecInt       = gpuistl::GpuVector<int>;

    using GpuSystemSeqOp  = GpuSystemSeqOpT<Scalar>;
    using OuterSolverT    = Dune::FlexibleSolver<GpuSystemSeqOp>;
    using SysPrecondT     = Dune::PreconditionerWithUpdate<
                                GpuSystemVectorT<Scalar>,
                                GpuSystemVectorT<Scalar>>;

    constexpr static std::size_t pressureIndex
        = Indices::pressureSwitchIdx;

    static_assert(Indices::numEq == numResDofs,
        "ISTLSolverGPUSystem supports only 3-phase blackoil models (numEq==3).");

public:

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    ISTLSolverGPUSystem(const Simulator&                  simulator,
                        const FlowLinearSolverParameters& params,
                        bool                              forceSerial = false)
        : simulator_(simulator)
        , params_(params)
        , forceSerial_(forceSerial)
        , elementChunks_(simulator.vanguard().gridView(),
                         Dune::Partitions::all,
                         ThreadManager::maxThreads())
    {
#if HAVE_MPI
        comm_ = std::make_shared<CommunicationType>(
            simulator.vanguard().grid().comm());
        extractParallelGridInformationToISTL(
            simulator.vanguard().grid(), m_parallelInformation_);
        {
            using ElementMapper = GetPropType<TypeTag, Properties::ElementMapper>;
            ElementMapper elemMapper(simulator.vanguard().gridView(),
                                     Dune::mcmgElementLayout());
            Opm::detail::findOverlapAndInterior(
                simulator.vanguard().grid(), elemMapper,
                overlapRows_, interiorRows_);
        }
        if (comm_->communicator().size() > 1) {
            const std::size_t size
                = simulator.vanguard().grid().leafGridView().size(0);
            Opm::detail::copyParValues(m_parallelInformation_, size, *comm_);
        }
#else
        comm_ = std::make_shared<CommunicationType>(
            simulator.gridView().comm());
#endif

        params_.init(
            simulator.vanguard().eclState().getSimulationConfig().useCPR());
        prm_ = setupPropertyTree(
            params_,
            Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
            Parameters::IsSet<Parameters::LinearSolverReduction>());

        if (Parameters::Get<Parameters::MatrixAddWellContributions>()) {
            OPM_THROW(std::logic_error,
                "ISTLSolverGPUSystem requires --matrix-add-well-contributions=false. "
                "Well contributions must be passed separately via the well model.");
        }
    }

    explicit ISTLSolverGPUSystem(const Simulator& simulator)
        : ISTLSolverGPUSystem(simulator, FlowLinearSolverParameters(), false)
    {}

    // ------------------------------------------------------------------
    // AbstractISTLSolver interface
    // ------------------------------------------------------------------

    void eraseMatrix() override {}

    void setActiveSolver(int num) override
    {
        if (num != 0)
            OPM_THROW(std::logic_error, "Only one solver available for ISTLSolverGPUSystem.");
    }

    int numAvailableSolvers() const override { return 1; }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        prepare(M.istlMatrix(), b);
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        prepareGpuSystemSolver(M, b);
    }

    void setResidual(Vector&) override {}

    void getResidual(Vector& b) const override
    {
        if (gpuRhs_)
            gpuRhs_->copyToHost(b);
    }

    void setMatrix(const SparseMatrixAdapter&) override {}

    // ------------------------------------------------------------------
    // solve — run the outer GPU Krylov solver.
    // ------------------------------------------------------------------
    bool solve(Vector& x, SimulatorReportSingle* report_ptr) override
    {
        ++solveCount_;

        // Forward report pointer to the system preconditioner for timing.
        if (auto* p = dynamic_cast<SystemPreconditioner<GpuSystemBackend<Scalar>>*>(sysPrecond_))
            p->setSimulatorReportPointer(report_ptr);

        const std::size_t nRes  = gpuA_->N() * numResDofs;
        const std::size_t nWell = gpuD_->N() * numWellDofs;

        // Build GPU system RHS.
        // Reservoir part: already uploaded during prepare().
        // Well part: upload from CPU merged residual.
        //
        // NOTE: GpuVector::operator=(const GpuVector&) does NOT resize — it only
        // memcpy-s when both sides are non-zero and the same size.  Use emplace()
        // to call the GpuSystemVectorT(nRes, nWell) constructor directly, which
        // allocates fresh GPU memory of the right size.  Re-emplace whenever dims
        // change (e.g., after a well-count change).
        if (!sysRhs_ || sysRhs_->res.dim() != nRes || sysRhs_->well.dim() != nWell)
            sysRhs_.emplace(nRes, nWell);
        sysRhs_->res = *gpuRhs_;
        sysRhs_->well.copyFromHost(mergedWellResidual_);

        // Zero-initialise solution vector.
        if (!sysX_ || sysX_->res.dim() != nRes || sysX_->well.dim() != nWell)
            sysX_.emplace(nRes, nWell);
        *sysX_ = Scalar(0);

        // Outer Krylov solve.
        Dune::InverseOperatorResult result;
        sysSolver_->apply(*sysX_, *sysRhs_, result);
        lastSeenIterations_ = result.iterations;

        // Copy reservoir solution GPU → CPU.
        sysX_->res.copyToHost(x);

        // Copy well solution GPU → CPU for getWellSolution().
        cpuWellSolution_.resize(gpuD_->N());
        sysX_->well.copyToHost(cpuWellSolution_);

        return checkConvergence(result);
    }

    int iterations() const override { return lastSeenIterations_; }

    const CommunicationType* comm() const override { return comm_.get(); }

    int getSolveCount() const override { return solveCount_; }

    std::optional<typename Parent::WellSolutionView>
    getWellSolution() const override
    {
        if (cpuWellSolution_.N() == 0 || wellDofOffsets_.empty())
            return std::nullopt;
        return typename Parent::WellSolutionView{
            cpuWellSolution_, wellDofOffsets_};
    }

private:

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------

    // Simulator reference + config
    const Simulator&              simulator_;
    FlowLinearSolverParameters    params_;
    PropertyTree                  prm_;
    std::shared_ptr<CommunicationType> comm_;
    bool                          forceSerial_;
    ElementChunksType             elementChunks_;

    // GPU reservoir matrix and RHS (set during prepare())
    std::optional<GpuSparseMatrix>                     gpuA_;
    std::optional<GpuVec>                              gpuRhs_;

    // GPU well matrices (B=WR, C=RW, D=WW)
    // B and C have non-square blocks (4×3 and 3×4 respectively), so they
    // are stored as scalar CSR matrices (GpuSparseMatrixGeneric, blockSize=1).
    // D has square 4×4 blocks and uses the standard GpuSparseMatrix wrapper.
    std::optional<gpuistl::GpuSparseMatrixGeneric<Scalar>> gpuB_;
    std::optional<gpuistl::GpuSparseMatrixGeneric<Scalar>> gpuC_;
    std::optional<GpuSparseMatrix>                         gpuD_;

    // CPU merged well matrices — owned storage; GpuSystemMatrixT points here.
    WRMatrixT<Scalar>   mergedB_;
    RWMatrixT<Scalar>   mergedC_;
    WWMatrixT<Scalar>   mergedD_;
    WellVectorT<Scalar> mergedWellResidual_;
    std::vector<int>    wellDofOffsets_;
    WellVectorT<Scalar> cpuWellSolution_;

    // GPU system matrix view
    GpuSystemMatrixT<Scalar>  sysMatrix_;

    // GPU Krylov work vectors — stored as optional so they can be created via
    // emplace(nRes, nWell), bypassing GpuVector::operator=(const GpuVector&)
    // which does NOT resize a 0-element vector.
    std::optional<GpuSystemVectorT<Scalar>> sysX_;
    std::optional<GpuSystemVectorT<Scalar>> sysRhs_;

    // Outer GPU Krylov solver
    std::unique_ptr<GpuSystemSeqOp> sysOp_;
    std::unique_ptr<OuterSolverT>   sysSolver_;
    SysPrecondT*                    sysPrecond_ = nullptr;

    // CPR weights for the reservoir sub-solver
    Vector             cpuWeights_;
    std::optional<GpuVec>    gpuWeights_;
    std::optional<GpuVecInt> gpuDiagIdx_;

    // State
    bool        sysInitialized_      = false;
    std::size_t cachedWellDofs_      = 0;
    int         lastSeenIterations_  = 0;
    int         solveCount_          = 0;

    // MPI helpers (only used when HAVE_MPI)
    std::any              m_parallelInformation_;
    std::vector<int>      overlapRows_;
    std::vector<int>      interiorRows_;

    // ------------------------------------------------------------------
    // checkConvergence — delegates to AbstractISTLSolver's static method.
    // ------------------------------------------------------------------
    bool checkConvergence(const Dune::InverseOperatorResult& result) const
    {
        return Parent::checkConvergence(result, params_);
    }

    // ------------------------------------------------------------------
    // prepareGpuSystemSolver — called by prepare(); orchestrates the full
    // GPU system setup on each Newton step.
    // ------------------------------------------------------------------
    void prepareGpuSystemSolver(const Matrix& M, Vector& b)
    {
        // 1. Upload reservoir matrix to GPU.
        if (!gpuA_) {
            gpuA_.emplace(GpuSparseMatrix::fromMatrix(M));
        } else {
            gpuA_->updateNonzeroValues(M, /*synchronize=*/true);
        }

        // 2. Upload reservoir RHS to GPU.
        if (!gpuRhs_) {
            gpuRhs_.emplace(b);
        } else {
            gpuRhs_->copyFromHostAsync(b);
        }

        // 3. Collect and merge per-well B/C/D matrices on CPU.
        std::vector<WRMatrixT<Scalar>> b_matrices;
        std::vector<RWMatrixT<Scalar>> c_matrices;
        std::vector<WWMatrixT<Scalar>> d_matrices;
        std::vector<std::vector<int>>  wcells;
        std::vector<WellVectorT<Scalar>> well_residuals;

        simulator_.problem().wellModel().addBCDMatrix(
            b_matrices, c_matrices, d_matrices, wcells, well_residuals);

        WellMatrixMerger<Scalar> merger(M.N());
        for (std::size_t i = 0; i < b_matrices.size(); ++i) {
            merger.addWell(b_matrices[i], c_matrices[i], d_matrices[i],
                           wcells[i], static_cast<int>(i),
                           "Well" + std::to_string(i + 1), well_residuals[i]);
        }
        merger.finalize();

        mergedWellResidual_ = std::move(merger.getMergedWellResidual());
        wellDofOffsets_     = std::move(merger.getWellDofOffsets());

        const std::size_t newWellDofs = merger.getMergedD().N();
        const bool needRebuild = !sysInitialized_ || (newWellDofs != cachedWellDofs_);

        mergedB_ = std::move(merger.getMergedB());
        mergedC_ = std::move(merger.getMergedC());
        mergedD_ = std::move(merger.getMergedD());
        cachedWellDofs_ = newWellDofs;

        // 4. Upload well matrices to GPU.
        //
        // B (WR, 4×3 blocks) and C (RW, 3×4 blocks) have non-square blocks that
        // are unsupported by GpuSparseMatrix (BSR, requires square blocks).  They
        // are always rebuilt from a scalar CSR expansion via makeNonSquareScalarGpuMatrix.
        // D (WW, 4×4 blocks) is square and uses the standard incremental update path.
        gpuB_.emplace(makeNonSquareScalarGpuMatrix<Scalar, numWellDofs, numResDofs>(mergedB_));
        gpuC_.emplace(makeNonSquareScalarGpuMatrix<Scalar, numResDofs, numWellDofs>(mergedC_));

        if (needRebuild || !gpuD_) {
            gpuD_.emplace(GpuSparseMatrix::fromMatrix(mergedD_));
        } else {
            gpuD_->updateNonzeroValues(mergedD_, /*synchronize=*/true);
        }

        // 5. Wire system matrix block pointers.
        sysMatrix_.A    = &*gpuA_;
        sysMatrix_.B    = &*gpuB_;
        sysMatrix_.C    = &*gpuC_;
        sysMatrix_.D    = &*gpuD_;
        sysMatrix_.cpuD = &mergedD_;

        // 6. Create outer solver on first call / well-dof change; update otherwise.
        if (needRebuild) {
            createGpuSystemSolver();
            sysInitialized_ = true;
        } else {
            sysPrecond_->update();
        }
    }

    // ------------------------------------------------------------------
    // makeWeightsCalculator — returns a std::function that computes CPR
    // weights for the reservoir block A and packages them as the .res
    // part of a GpuSystemVectorT (the .well part is left empty).
    //
    // Called during createGpuSystemSolver(); gpuA_ must be valid.
    // ------------------------------------------------------------------
    std::function<GpuSystemVectorT<Scalar>()>
    makeWeightsCalculator(const PropertyTree& resSolverPrm)
    {
        using namespace std::string_literals;
        const auto weightsType
            = resSolverPrm.get("preconditioner.weight_type"s, "quasiimpes"s);

        if (weightsType == "quasiimpes") {
            const std::size_t nWeights = gpuA_->N() * numResDofs;
            gpuWeights_.emplace(nWeights);
            auto diagIdx = Amg::precomputeDiagonalIndices(*gpuA_);
            gpuDiagIdx_.emplace(diagIdx);

            return [this]() -> GpuSystemVectorT<Scalar> {
                Amg::getQuasiImpesWeights<Scalar, /*transpose=*/false>(
                    *gpuA_, pressureIndex, *gpuWeights_, *gpuDiagIdx_);
                // Return a GpuSystemVectorT with .res = copy of weights.
                // .well is left as a 0-element vector; the factory only uses .res.
                GpuSystemVectorT<Scalar> w(gpuWeights_->dim(), 0);
                w.res = *gpuWeights_;
                return w;
            };

        } else if (weightsType == "trueimpes") {
            cpuWeights_.resize(gpuA_->N());
            gpuWeights_.emplace(cpuWeights_);
            const bool threadParallel = params_.cpr_weights_thread_parallel_;

            return [this, threadParallel]() -> GpuSystemVectorT<Scalar> {
                ElementContext elemCtx(simulator_);
                Amg::getTrueImpesWeights(pressureIndex, cpuWeights_, elemCtx,
                                          simulator_.model(), elementChunks_,
                                          threadParallel);
                gpuWeights_->copyFromHostAsync(cpuWeights_);
                GpuSystemVectorT<Scalar> w(gpuWeights_->dim(), 0);
                w.res = *gpuWeights_;
                return w;
            };

        } else if (weightsType == "trueimpesanalytic") {
            cpuWeights_.resize(gpuA_->N());
            gpuWeights_.emplace(cpuWeights_);
            const bool threadParallel = params_.cpr_weights_thread_parallel_;

            return [this, threadParallel]() -> GpuSystemVectorT<Scalar> {
                ElementContext elemCtx(simulator_);
                Amg::getTrueImpesWeightsAnalytic(pressureIndex, cpuWeights_, elemCtx,
                                                  simulator_.model(), elementChunks_,
                                                  threadParallel);
                gpuWeights_->copyFromHostAsync(cpuWeights_);
                GpuSystemVectorT<Scalar> w(gpuWeights_->dim(), 0);
                w.res = *gpuWeights_;
                return w;
            };

        } else {
            // No weight computation — pass empty function.
            return {};
        }
    }

    // ------------------------------------------------------------------
    // createGpuSystemSolver — builds the outer FlexibleSolver from scratch.
    // ------------------------------------------------------------------
    void createGpuSystemSolver()
    {
        auto resSolverPrm = prm_.get_child("preconditioner.reservoir_solver");
        auto sysWeightCalc = makeWeightsCalculator(resSolverPrm);

        sysOp_     = std::make_unique<GpuSystemSeqOp>(sysMatrix_);
        sysSolver_ = std::make_unique<OuterSolverT>(
            *sysOp_, prm_, sysWeightCalc, pressureIndex);
        sysPrecond_ = &sysSolver_->preconditioner();
    }
};

} // namespace Opm::gpusystem
