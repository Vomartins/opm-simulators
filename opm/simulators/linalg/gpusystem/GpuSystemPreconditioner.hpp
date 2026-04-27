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

#include <opm/simulators/linalg/gpusystem/GpuSystemTypes.hpp>

#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/detail/FlexibleSolverWrapper.hpp>
#else
#include <opm/simulators/linalg/gpuistl/detail/FlexibleSolverWrapper.hpp>
#endif

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <dune/common/parallel/communication.hh>
#include <dune/common/timer.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>

#include <functional>
#include <memory>

namespace Opm::gpusystem
{

// --------------------------------------------------------------------------
// GpuSystemPreconditioner
//
// GPU analogue of SystemPreconditioner.  Implements the same 3-stage
// reservoir-well algorithm but with GPU work vectors:
//
//   Stage 1 — Reservoir CPR solve (GPU FlexibleSolverWrapper on A block)
//   Stage 2 — Well solve (CPU round-trip) + reservoir smoother (GPU)
//   Stage 3 — Final well solve (CPU round-trip)
//
// The well sub-solver executes on CPU because UMFPACK/DILU/ILU0 are not yet
// GPU-native.  Well systems are small (n_w << n_r), so the PCIe transfer
// overhead is negligible.
//
// Sequential only: no parallel communicator template parameter.
//
// Timing is recorded via Dune::Timer.  For accurate GPU stage timing the
// caller should issue cudaDeviceSynchronize() before reading the report, or
// CUDA events should replace Dune::Timer in a future revision.
// --------------------------------------------------------------------------
template<typename Scalar>
class GpuSystemPreconditioner
    : public Dune::PreconditionerWithUpdate<GpuSystemVectorT<Scalar>, GpuSystemVectorT<Scalar>>
{
public:
    // GPU reservoir sub-solver type (serial communicator).
    using SerialComm      = Dune::Communication<int>;
    using GpuResMatrixT   = GpuRRMatrixT<Scalar>;
    using GpuVec          = gpuistl::GpuVector<Scalar>;
    using ResSolverT      = gpuistl::detail::FlexibleSolverWrapper<GpuResMatrixT, GpuVec, SerialComm>;

    // CPU well sub-solver types.
    using WellOperator    = Dune::MatrixAdapter<WWMatrixT<Scalar>, WellVectorT<Scalar>, WellVectorT<Scalar>>;
    using WellSolverT     = Dune::FlexibleSolver<WellOperator>;

    // Weight calculator function type — identical to ISTLSolverGPUISTL convention.
    using WeightsCalcT    = std::function<GpuVec()>;

    SimulatorReportSingle* report_ptr_ = nullptr;

    void setSimulatorReportPointer(SimulatorReportSingle* p) { report_ptr_ = p; }
    SimulatorReportSingle* simulatorReportPointer() const    { return report_ptr_; }

    // -----------------------------------------------------------------------
    // Constructor
    //
    // \param S               GPU system matrix view (holds const pointers to
    //                        the four GPU sub-blocks; must outlive this object)
    // \param cpuD            CPU copy of the well-well block D, used to
    //                        build the CPU well sub-solver
    // \param weightsCalc     Returns a GpuVector of CPR weights for the
    //                        reservoir pressure variable (same convention as
    //                        ISTLSolverGPUISTL::getWeightsCalculator())
    // \param pressureIndex   Column index of the pressure unknown
    // \param prm             Solver configuration; must have sub-trees
    //                        "reservoir_solver", "reservoir_smoother", and
    //                        "well_solver" (same JSON layout as SystemCPR)
    // -----------------------------------------------------------------------
    GpuSystemPreconditioner(const GpuSystemMatrixT<Scalar>& S,
                            const WWMatrixT<Scalar>&         cpuD,
                            WeightsCalcT                     weightsCalc,
                            int                              pressureIndex,
                            const PropertyTree&              prm)
        : S_(S)
    {
        initSubSolvers(cpuD, std::move(weightsCalc), pressureIndex, prm);
        initWorkVectors();
    }

    // Dune::Preconditioner interface — required by PreconditionerWithUpdate.
    void pre(GpuSystemVectorT<Scalar>&, GpuSystemVectorT<Scalar>&) override {}
    void post(GpuSystemVectorT<Scalar>&)                             override {}
    Dune::SolverCategory::Category category() const override
    { return Dune::SolverCategory::sequential; }
    bool hasPerfectUpdate() const override { return false; }

    // -----------------------------------------------------------------------
    // update — propagates non-zero value changes to all sub-solvers.
    // Call after ISTLSolverGPUSystem has uploaded fresh non-zero values to
    // the GPU sub-blocks.
    // -----------------------------------------------------------------------
    void update() override
    {
        resSolver_ ->update();
        resSmoother_->update();
        wellSolver_ ->preconditioner().update();
    }

    // -----------------------------------------------------------------------
    // apply — runs the 3-stage preconditioner.
    //
    // \param v   Output: approximate solution increment
    // \param d   Input:  defect / right-hand side (not modified)
    // -----------------------------------------------------------------------
    void apply(GpuSystemVectorT<Scalar>& v, const GpuSystemVectorT<Scalar>& d) override
    {
        const auto& A = S_.getRR();   // reservoir–reservoir
        const auto& C = S_.getRW();   // reservoir–well coupling
        const auto& B = S_.getWR();   // well–reservoir coupling
        const auto& D = S_.getWW();   // well–well

        // Initialise residuals and solutions.
        // (All work vectors are std::optional<GpuVec>; dereference with *.)
        *resRes_  = d.res;
        *wRes_    = d.well;
        *resSol_  = Scalar(0);
        *wSol_    = Scalar(0);

        Dune::Timer stage_timer;

        // -------------------------------------------------------------------
        // Stage 1: Reservoir CPR solve
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            Dune::InverseOperatorResult res_result;
            *dresSol_    = Scalar(0);
            *tmp_resRes_ = *resRes_;
            resSolver_->apply(*dresSol_, *tmp_resRes_, res_result);
            *resSol_ += *dresSol_;
            A.usmv(Scalar(-1), *dresSol_, *resRes_);   // resRes_ -= A * dresSol_
            B.usmv(Scalar(-1), *dresSol_, *wRes_);     // wRes_   -= B * dresSol_
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage1_time += stage_timer.lastElapsed();

        // -------------------------------------------------------------------
        // Stage 2a: Well solve (CPU round-trip)
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            Dune::InverseOperatorResult well_result;
            *dwSol_ = Scalar(0);
            applyWellSolverCpu(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
            C.usmv(Scalar(-1), *dwSol_, *resRes_);   // resRes_ -= C * dwSol_
            D.usmv(Scalar(-1), *dwSol_, *wRes_);     // wRes_   -= D * dwSol_
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage2_well_time += stage_timer.lastElapsed();

        // -------------------------------------------------------------------
        // Stage 2b: Reservoir smoother
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            Dune::InverseOperatorResult res_result;
            *dresSol_    = Scalar(0);
            *tmp_resRes_ = *resRes_;
            resSmoother_->apply(*dresSol_, *tmp_resRes_, res_result);
            *resSol_ += *dresSol_;
            B.usmv(Scalar(-1), *dresSol_, *wRes_);   // wRes_ -= B * dresSol_
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage2_res_time += stage_timer.lastElapsed();

        // -------------------------------------------------------------------
        // Stage 3: Final well solve (CPU round-trip)
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            *dwSol_ = Scalar(0);
            applyWellSolverCpu(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage3_time += stage_timer.lastElapsed();

        v.res  = *resSol_;
        v.well = *wSol_;
    }

private:
    const GpuSystemMatrixT<Scalar>& S_;

    // GPU reservoir sub-solvers.
    std::unique_ptr<ResSolverT>  resSolver_;
    std::unique_ptr<ResSolverT>  resSmoother_;

    // CPU well sub-solver.
    std::unique_ptr<WellOperator> wop_;
    std::unique_ptr<WellSolverT>  wellSolver_;

    // GPU work vectors (flat scalar buffers).
    // Stored as optional so they can be constructed in-place via emplace(n),
    // bypassing GpuVector::operator=(const GpuVector&) which does NOT resize.
    std::optional<GpuVec> resRes_;     // current reservoir residual
    std::optional<GpuVec> resSol_;     // accumulated reservoir solution
    std::optional<GpuVec> dresSol_;    // reservoir solution increment
    std::optional<GpuVec> tmp_resRes_; // scratch copy of resRes_ passed to sub-solvers
    std::optional<GpuVec> wRes_;       // current well residual
    std::optional<GpuVec> wSol_;       // accumulated well solution
    std::optional<GpuVec> dwSol_;      // well solution increment

    // CPU-side buffers for the well solver round-trip.
    WellVectorT<Scalar> cpuWRes_;
    WellVectorT<Scalar> cpuDwSol_;

    // -------------------------------------------------------------------
    // applyWellSolverCpu
    //
    // Executes the CPU well solver for one stage:
    //   1. Copy the GPU well-residual (gpuRhs) to CPU.
    //   2. Apply the CPU FlexibleSolver: cpuDwSol_ = D^{-1}_approx * cpuWRes_
    //   3. Copy the CPU result back to the GPU output (gpuSol).
    //
    // Both gpuSol and gpuRhs must have size nWellRows * numWellDofs.
    // -------------------------------------------------------------------
    void applyWellSolverCpu(GpuVec& gpuSol, const GpuVec& gpuRhs)
    {
        // device → host
        gpuRhs.copyToHost(cpuWRes_);

        Dune::InverseOperatorResult well_result;
        cpuDwSol_ = Scalar(0);
        wellSolver_->apply(cpuDwSol_, cpuWRes_, well_result);

        // host → device
        gpuSol.copyFromHost(cpuDwSol_);
    }

    // -------------------------------------------------------------------
    // initSubSolvers — constructs reservoir (GPU) and well (CPU) solvers.
    // -------------------------------------------------------------------
    void initSubSolvers(const WWMatrixT<Scalar>& cpuD,
                        WeightsCalcT             weightsCalc,
                        int                      pressureIndex,
                        const PropertyTree&      prm)
    {
        const auto& resA = S_.getRR();

        auto resprm         = prm.get_child("reservoir_solver");
        auto resprmsmoother = prm.get_child("reservoir_smoother");
        auto wellprm        = prm.get_child("well_solver");

        // Reservoir sub-solvers — serial, GPU.
        // FlexibleSolverWrapper(matrix, parallel, prm, pressureIndex,
        //                       weightsCalc, forceSerial, comm)
        resSolver_ = std::make_unique<ResSolverT>(
            resA, /*parallel=*/false, resprm,
            pressureIndex, weightsCalc, /*forceSerial=*/true, /*comm=*/nullptr);

        resSmoother_ = std::make_unique<ResSolverT>(
            resA, /*parallel=*/false, resprmsmoother,
            pressureIndex, weightsCalc, /*forceSerial=*/true, /*comm=*/nullptr);

        // Well sub-solver — CPU.
        std::function<WellVectorT<Scalar>()> noWeights;
        wop_ = std::make_unique<WellOperator>(cpuD);
        wellSolver_ = std::make_unique<WellSolverT>(
            *wop_, wellprm, noWeights, pressureIndex);
    }

    // -------------------------------------------------------------------
    // initWorkVectors — allocates GPU and CPU work buffers.
    //
    // All GPU vectors are flat scalar buffers:
    //   reservoir vectors — nResRows * numResDofs elements
    //   well vectors      — nWellRows * numWellDofs elements
    // -------------------------------------------------------------------
    void initWorkVectors()
    {
        const std::size_t nRes  = S_.getRR().N() * numResDofs;
        const std::size_t nWell = S_.getWW().N() * numWellDofs;

        // Use emplace() to call the GpuVector(size_t) constructor directly.
        // GpuVector::operator=(const GpuVector&) does NOT resize a 0-element vector,
        // so plain assignment (e.g. resRes_ = GpuVec(nRes)) would be silently ignored.
        resRes_    .emplace(nRes);
        resSol_    .emplace(nRes);
        dresSol_   .emplace(nRes);
        tmp_resRes_.emplace(nRes);
        wRes_      .emplace(nWell);
        wSol_      .emplace(nWell);
        dwSol_     .emplace(nWell);

        cpuWRes_ .resize(S_.getWW().N());
        cpuDwSol_.resize(S_.getWW().N());
    }
};

} // namespace Opm::gpusystem
