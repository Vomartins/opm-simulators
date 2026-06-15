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
#include <opm/simulators/linalg/gpuistl_hip/PinnedMemoryHolder.hpp>
#else
#include <opm/simulators/linalg/gpuistl/detail/FlexibleSolverWrapper.hpp>
#include <opm/simulators/linalg/gpuistl/PinnedMemoryHolder.hpp>
#endif

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <dune/common/parallel/communication.hh>
#include <dune/common/timer.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>

#include <hip/hip_runtime.h>

#include <cmath>
#include <functional>
#include <memory>
#include <string>

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
// Timing is recorded via hipEvent pairs for accurate GPU-side measurement
// and Dune::Timer for CPU wall-clock time.  The report receives GPU time
// (hipEventElapsedTime) which reflects actual device execution.
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
        hipEventCreate(&evStart_);
        hipEventCreate(&evStop_);
        hipEventCreate(&evXferStart_);
        hipEventCreate(&evXferStop_);
        initSubSolvers(cpuD, std::move(weightsCalc), pressureIndex, prm);
        initWorkVectors();
    }

    ~GpuSystemPreconditioner() override
    {
        hipEventDestroy(evStart_);
        hipEventDestroy(evStop_);
        hipEventDestroy(evXferStart_);
        hipEventDestroy(evXferStop_);
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

        // Stage timing uses hipEvent pairs for accurate GPU-side
        // measurement and Dune::Timer for CPU wall-clock time.
        // hipEventElapsedTime() returns milliseconds; we convert to seconds
        // for consistency with Dune::Timer and SimulatorReportSingle.
        Dune::Timer stage_timer;
        float gpu_ms = 0.0f;

        // -------------------------------------------------------------------
        // Stage 1: Reservoir CPR solve
        // -------------------------------------------------------------------
        hipEventRecord(evStart_);
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
        hipEventRecord(evStop_);
        hipEventSynchronize(evStop_);
        stage_timer.stop();
        if (report_ptr_) {
            hipEventElapsedTime(&gpu_ms, evStart_, evStop_);
            report_ptr_->sys_stage1_time += static_cast<double>(gpu_ms) * 1e-3;
        }

        // -------------------------------------------------------------------
        // Stage 2a: Well solve (CPU round-trip)
        // -------------------------------------------------------------------
        hipEventRecord(evStart_);
        stage_timer.start();
        {
            Dune::InverseOperatorResult well_result;
            *dwSol_ = Scalar(0);
            applyWellSolverCpu(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
            C.usmv(Scalar(-1), *dwSol_, *resRes_);   // resRes_ -= C * dwSol_
            D.usmv(Scalar(-1), *dwSol_, *wRes_);     // wRes_   -= D * dwSol_
        }
        hipEventRecord(evStop_);
        hipEventSynchronize(evStop_);
        stage_timer.stop();
        if (report_ptr_) {
            hipEventElapsedTime(&gpu_ms, evStart_, evStop_);
            report_ptr_->sys_stage2_well_time += static_cast<double>(gpu_ms) * 1e-3;
        }

        // -------------------------------------------------------------------
        // Stage 2b: Reservoir smoother
        // -------------------------------------------------------------------
        hipEventRecord(evStart_);
        stage_timer.start();
        {
            Dune::InverseOperatorResult res_result;
            *dresSol_    = Scalar(0);
            *tmp_resRes_ = *resRes_;
            resSmoother_->apply(*dresSol_, *tmp_resRes_, res_result);
            *resSol_ += *dresSol_;
            B.usmv(Scalar(-1), *dresSol_, *wRes_);   // wRes_ -= B * dresSol_
            A.usmv(Scalar(-1), *dresSol_, *resRes_); // Update residuals for debugging
        }
        hipEventRecord(evStop_);
        hipEventSynchronize(evStop_);
        stage_timer.stop();
        if (report_ptr_) {
            hipEventElapsedTime(&gpu_ms, evStart_, evStop_);
            report_ptr_->sys_stage2_res_time += static_cast<double>(gpu_ms) * 1e-3;
        }

        // -------------------------------------------------------------------
        // Stage 3: Final well solve (CPU round-trip)
        // -------------------------------------------------------------------
        hipEventRecord(evStart_);
        stage_timer.start();
        {
            *dwSol_ = Scalar(0);
            applyWellSolverCpu(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
            C.usmv(Scalar(-1), *dwSol_, *resRes_); // Update residuals for debugging
            D.usmv(Scalar(-1), *dwSol_, *wRes_); // Update residuals for debugging
        }
        hipEventRecord(evStop_);
        hipEventSynchronize(evStop_);
        stage_timer.stop();
        if (report_ptr_) {
            hipEventElapsedTime(&gpu_ms, evStart_, evStop_);
            report_ptr_->sys_stage3_time += static_cast<double>(gpu_ms) * 1e-3;
        }

        // -------------------------------------------------------------------
        // NaN guard: check solution norms before returning.
        // If any stage produced NaN/Inf, zero the output so GMRES sees no
        // correction and the Newton solver can cut the timestep gracefully
        // instead of propagating NaN into the next iteration.
        // -------------------------------------------------------------------
        const Scalar resNorm = resSol_->two_norm();
        const Scalar wellNorm = wSol_->two_norm();

        if (std::isnan(resNorm) || std::isinf(resNorm)
            || std::isnan(wellNorm) || std::isinf(wellNorm)) {
            OpmLog::warning("GpuSystemPreconditioner: NaN/Inf detected in "
                            "preconditioner output (resNorm=" + std::to_string(resNorm)
                            + ", wellNorm=" + std::to_string(wellNorm)
                            + "). Zeroing solution to allow graceful timestep cut.");
            *resSol_ = Scalar(0);
            *wSol_   = Scalar(0);
        }

        v.res  = *resSol_;
        v.well = *wSol_;
    }

private:
    const GpuSystemMatrixT<Scalar>& S_;

    // HIP events for GPU-side stage timing.
    hipEvent_t evStart_;
    hipEvent_t evStop_;

    // HIP events for well-solver transfer/solve timing.
    hipEvent_t evXferStart_;
    hipEvent_t evXferStop_;

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
    // Declared before pinned holders so that unpin (holder destruction)
    // happens before the underlying Dune vectors are freed.
    WellVectorT<Scalar> cpuWRes_;
    WellVectorT<Scalar> cpuDwSol_;

    // Pinned (page-locked) registration for the CPU well buffers.
    // Pinning enables the HIP runtime to use DMA for host↔device copies,
    // avoiding an extra staging buffer and reducing PCIe transfer latency.
    std::optional<gpuistl::PinnedMemoryHolder<Scalar>> pinnedWRes_;
    std::optional<gpuistl::PinnedMemoryHolder<Scalar>> pinnedDwSol_;

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
        float ms = 0.0f;

        // device → host
        hipEventRecord(evXferStart_);
        gpuRhs.copyToHost(cpuWRes_);
        hipEventRecord(evXferStop_);
        hipEventSynchronize(evXferStop_);
        hipEventElapsedTime(&ms, evXferStart_, evXferStop_);
        double d2h_s = static_cast<double>(ms) * 1e-3;

        // CPU well solve
        hipEventRecord(evXferStart_);
        Dune::InverseOperatorResult well_result;
        cpuDwSol_ = Scalar(0);
        wellSolver_->apply(cpuDwSol_, cpuWRes_, well_result);
        hipEventRecord(evXferStop_);
        hipEventSynchronize(evXferStop_);
        hipEventElapsedTime(&ms, evXferStart_, evXferStop_);
        double cpu_s = static_cast<double>(ms) * 1e-3;

        // host → device
        hipEventRecord(evXferStart_);
        gpuSol.copyFromHost(cpuDwSol_);
        hipEventRecord(evXferStop_);
        hipEventSynchronize(evXferStop_);
        hipEventElapsedTime(&ms, evXferStart_, evXferStop_);
        double h2d_s = static_cast<double>(ms) * 1e-3;

        std::cout << "[SysCPR xfer] D2H=" << d2h_s*1e3
                  << "ms  CPU_solve=" << cpu_s*1e3
                  << "ms  H2D=" << h2d_s*1e3 << "ms\n";

        if (report_ptr_) {
            report_ptr_->sys_d2h_time       += d2h_s;
            report_ptr_->sys_cpu_solve_time  += cpu_s;
            report_ptr_->sys_h2d_time        += h2d_s;
        }
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

        // Pin the CPU well buffers so HIP can use DMA for host↔device copies.
        // The underlying Dune::BlockVector storage is contiguous (verified by
        // GpuVector::copyToHost/copyFromHost which use &bvector[0][0]).
        if (nWell > 0) {
            pinnedWRes_ .emplace(&cpuWRes_ [0][0], nWell);
            pinnedDwSol_.emplace(&cpuDwSol_[0][0], nWell);
        }
    }
};

} // namespace Opm::gpusystem
