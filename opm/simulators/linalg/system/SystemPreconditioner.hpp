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
#ifndef OPM_SYSTEMPRECONDITIONER_HEADER_INCLUDED
#define OPM_SYSTEMPRECONDITIONER_HEADER_INCLUDED

#include <opm/simulators/linalg/system/SystemTypes.hpp>

#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <dune/common/timer.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solver.hh>

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

namespace Opm
{

struct SimulatorReportSingle;

// Reservoir operator/comm types used as template arguments.
template<typename Scalar>
using SeqResOperator = Dune::MatrixAdapter<RRMatrix<Scalar>, ResVector<Scalar>, ResVector<Scalar>>;

#if HAVE_MPI
using ParResComm = Dune::OwnerOverlapCopyCommunication<int, int>;
template<typename Scalar>
using ParResOperator = Dune::OverlappingSchwarzOperator<RRMatrix<Scalar>, ResVector<Scalar>, ResVector<Scalar>, ParResComm>;
#endif

// --------------------------------------------------------------------------
// SystemPreconditioner
//
// Unified preconditioner for the coupled reservoir-well system.
// Templated on a Backend traits struct that provides all type aliases and
// static helpers for either the CPU or GPU path.
//
// The 3-stage algorithm:
//   Stage 1 — Reservoir CPR solve
//   Stage 2 — Well solve + reservoir smoothing
//   Stage 3 — Final well solve
//
// For CPU parallel runs, copyOwnerToAll synchronises overlap DOFs before
// each reservoir sub-solve.
//
// For GPU runs, the well sub-solver executes on CPU with device-to-host
// round-trips (well systems are small, so PCIe overhead is negligible).
// --------------------------------------------------------------------------
template <class Backend>
class SystemPreconditioner
    : public Dune::PreconditionerWithUpdate<typename Backend::SysVector,
                                            typename Backend::SysVector>
{
    using Scalar       = typename Backend::ScalarType;
    using SysMatrix    = typename Backend::SysMatrix;
    using SysVector    = typename Backend::SysVector;
    using ResWork      = typename Backend::ResWork;
    using WellWork     = typename Backend::WellWork;
    using ResSolver    = typename Backend::ResSolver;
    using WellOp       = typename Backend::WellOperator;
    using WellSolver   = typename Backend::WellSolver;
    using WeightsCalc  = typename Backend::WeightsCalc;
    using ResCommType  = typename Backend::ResCommType;

public:
    static constexpr bool isParallel = Backend::isParallel;

    SimulatorReportSingle* report_ptr_ = nullptr;

    void setSimulatorReportPointer(SimulatorReportSingle* report) { report_ptr_ = report; }
    SimulatorReportSingle* simulatorReportPointer() const { return report_ptr_; }

    // -----------------------------------------------------------------------
    // Sequential constructor (all backends; also the only constructor for GPU).
    // -----------------------------------------------------------------------
    template <bool P = isParallel, std::enable_if_t<!P, int> = 0>
    SystemPreconditioner(const SysMatrix&     S,
                         WeightsCalc          weightsCalculator,
                         int                  pressureIndex,
                         const PropertyTree&  prm)
        : S_(S)
    {
        initSubSolvers(prm, std::move(weightsCalculator), pressureIndex);
        initWorkVectors();
    }

    // -----------------------------------------------------------------------
    // Parallel constructor (CPU only; enabled when Backend::isParallel).
    // -----------------------------------------------------------------------
    template <bool P = isParallel, std::enable_if_t<P, int> = 0>
    SystemPreconditioner(const SysMatrix&     S,
                         WeightsCalc          weightsCalculator,
                         int                  pressureIndex,
                         const PropertyTree&  prm,
                         const ResCommType&   resComm)
        : S_(S)
        , resComm_(&resComm)
    {
        initSubSolvers(prm, std::move(weightsCalculator), pressureIndex);
        initWorkVectors();
    }

    // Dune::Preconditioner interface.
    void pre(SysVector&, SysVector&) override {}
    void post(SysVector&) override {}

    Dune::SolverCategory::Category category() const override
    {
        if constexpr (isParallel)
            return Dune::SolverCategory::overlapping;
        else
            return Dune::SolverCategory::sequential;
    }

    bool hasPerfectUpdate() const override
    {
        return Backend::hasPerfectUpdate;
    }

    // -----------------------------------------------------------------------
    // update — propagates non-zero value changes to all sub-solvers.
    // -----------------------------------------------------------------------
    void update() override
    {
        Backend::updateResSolver(*resSolver_);
        Backend::updateResSolver(*resSmoother_);
        wellSolver_->preconditioner().update();
    }

    // -----------------------------------------------------------------------
    // apply — runs the 3-stage preconditioner.
    //
    // v  : Output — approximate solution increment
    // d  : Input  — defect / right-hand side (not modified)
    // -----------------------------------------------------------------------
    void apply(SysVector& v, const SysVector& d) override
    {
        // Sub-block references (named consistently with the matrix members:
        //   A = (0,0) reservoir-reservoir
        //   B = (1,0) well-reservoir
        //   C = (0,1) reservoir-well
        //   D = (1,1) well-well
        const auto& A = S_.getRR();
        const auto& B = S_.getWR();
        const auto& C = S_.getRW();
        const auto& D = S_.getWW();

        // Initialise residuals and solutions.
        *resRes_  = Backend::getRes(d);
        *wRes_    = Backend::getWell(d);
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
            syncResVector(*tmp_resRes_);
            resSolver_->apply(*dresSol_, *tmp_resRes_, res_result);
            *resSol_ += *dresSol_;
            A.usmv(Scalar(-1), *dresSol_, *resRes_);   // resRes_ -= A * dresSol_
            B.usmv(Scalar(-1), *dresSol_, *wRes_);     // wRes_   -= B * dresSol_
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage1_time += stage_timer.lastElapsed();

        // -------------------------------------------------------------------
        // Stage 2a: Well solve
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            *dwSol_ = Scalar(0);
            applyWellSolver(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
            C.usmv(Scalar(-1), *dwSol_, *resRes_);     // resRes_ -= C * dwSol_
            D.usmv(Scalar(-1), *dwSol_, *wRes_);       // wRes_   -= D * dwSol_
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
            syncResVector(*tmp_resRes_);
            resSmoother_->apply(*dresSol_, *tmp_resRes_, res_result);
            *resSol_ += *dresSol_;
            B.usmv(Scalar(-1), *dresSol_, *wRes_);     // wRes_ -= B * dresSol_
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage2_res_time += stage_timer.lastElapsed();

        // -------------------------------------------------------------------
        // Stage 3: Final well solve
        // -------------------------------------------------------------------
        stage_timer.start();
        {
            *dwSol_ = Scalar(0);
            applyWellSolver(*dwSol_, *wRes_);
            *wSol_ += *dwSol_;
        }
        stage_timer.stop();
        if (report_ptr_)
            report_ptr_->sys_stage3_time += stage_timer.lastElapsed();

        syncResVector(*resSol_);
        Backend::getRes(v)  = *resSol_;
        Backend::getWell(v) = *wSol_;
    }

private:
    const SysMatrix& S_;
    const ResCommType* resComm_ = nullptr;

    // Reservoir operator — only used for CPU backends (GPU FlexibleSolverWrapper
    // creates its own operator internally).
    std::unique_ptr<typename Backend::ResOperator> rop_;

    // Sub-solvers.
    std::unique_ptr<ResSolver>   resSolver_;
    std::unique_ptr<ResSolver>   resSmoother_;
    std::unique_ptr<WellOp>      wop_;
    std::unique_ptr<WellSolver>  wellSolver_;

    // GPU work vectors (flat scalar buffers) and CPU work vectors (block vectors)
    // are unified via std::optional.  Both ResWork(size_t) and WellWork(size_t)
    // constructors exist, so emplace(n) works for both backends.
    std::optional<ResWork>  resRes_;
    std::optional<ResWork>  resSol_;
    std::optional<ResWork>  dresSol_;
    std::optional<ResWork>  tmp_resRes_;
    std::optional<WellWork> wRes_;
    std::optional<WellWork> wSol_;
    std::optional<WellWork> dwSol_;

    // CPU well-solver scratch copy (CPU backend only — FlexibleSolver may modify RHS).
    std::optional<WellWork> tmp_wRes_;

    // CPU-side buffers for the GPU well-solver round-trip (GPU backend only).
    WellVectorT<Scalar> cpuWRes_;
    WellVectorT<Scalar> cpuDwSol_;

    // -----------------------------------------------------------------------
    // applyWellSolver — applies the well sub-solver.
    //
    // GPU: copies RHS to host, runs CPU FlexibleSolver, copies result back.
    // CPU: copies RHS to scratch, runs FlexibleSolver in-place.
    // -----------------------------------------------------------------------
    void applyWellSolver(WellWork& sol, WellWork& rhs)
    {
        if constexpr (Backend::isGpu) {
            rhs.copyToHost(cpuWRes_);
            Dune::InverseOperatorResult well_result;
            cpuDwSol_ = Scalar(0);
            wellSolver_->apply(cpuDwSol_, cpuWRes_, well_result);
            sol.copyFromHost(cpuDwSol_);
        } else {
            Dune::InverseOperatorResult well_result;
            *tmp_wRes_ = rhs;
            wellSolver_->apply(sol, *tmp_wRes_, well_result);
        }
    }

    // -----------------------------------------------------------------------
    // syncResVector — parallel overlap synchronisation (no-op for sequential
    // and GPU backends).
    // -----------------------------------------------------------------------
    void syncResVector(ResWork& v)
    {
        if constexpr (Backend::isParallel) {
            resComm_->copyOwnerToAll(v, v);
        }
    }

    // -----------------------------------------------------------------------
    // initSubSolvers — constructs reservoir and well sub-solvers.
    // -----------------------------------------------------------------------
    void initSubSolvers(const PropertyTree& prm,
                        WeightsCalc         weightsCalc,
                        int                 pressureIndex)
    {
        auto resprm         = prm.get_child("reservoir_solver");
        auto resprmsmoother = prm.get_child("reservoir_smoother");
        auto wellprm        = prm.get_child("well_solver");

        if constexpr (Backend::isGpu) {
            // GPU: FlexibleSolverWrapper creates its own operator from the matrix.
            const auto& resA = S_.getRR();
            resSolver_ = std::make_unique<ResSolver>(
                resA, /*parallel=*/false, resprm,
                pressureIndex, weightsCalc, /*forceSerial=*/true, /*comm=*/nullptr);
            resSmoother_ = std::make_unique<ResSolver>(
                resA, /*parallel=*/false, resprmsmoother,
                pressureIndex, weightsCalc, /*forceSerial=*/true, /*comm=*/nullptr);
        } else if constexpr (Backend::isParallel) {
            // CPU parallel: operator wraps the reservoir matrix + communicator.
            using ResOp = typename Backend::ResOperator;
            rop_ = std::make_unique<ResOp>(S_.getRR(), *resComm_);
            resSolver_ = std::make_unique<ResSolver>(
                *rop_, *resComm_, resprm, weightsCalc, pressureIndex);
            resSmoother_ = std::make_unique<ResSolver>(
                *rop_, *resComm_, resprmsmoother, weightsCalc, pressureIndex);
        } else {
            // CPU sequential.
            using ResOp = typename Backend::ResOperator;
            rop_ = std::make_unique<ResOp>(S_.getRR());
            resSolver_ = std::make_unique<ResSolver>(
                *rop_, resprm, weightsCalc, pressureIndex);
            resSmoother_ = std::make_unique<ResSolver>(
                *rop_, resprmsmoother, weightsCalc, pressureIndex);
        }

        // Well sub-solver — identical for all backends.
        const auto& cpuWellMatrix = Backend::getWellMatrixCpu(S_);
        wop_ = std::make_unique<WellOp>(cpuWellMatrix);
        std::function<WellVectorT<Scalar>()> noWeights;
        wellSolver_ = std::make_unique<WellSolver>(
            *wop_, wellprm, noWeights, pressureIndex);
    }

    // -----------------------------------------------------------------------
    // initWorkVectors — allocates GPU and CPU work buffers.
    // -----------------------------------------------------------------------
    void initWorkVectors()
    {
        const std::size_t nRes  = Backend::resWorkSize(S_);
        const std::size_t nWell = Backend::wellWorkSize(S_);

        resRes_    .emplace(nRes);
        resSol_    .emplace(nRes);
        dresSol_   .emplace(nRes);
        tmp_resRes_.emplace(nRes);
        wRes_      .emplace(nWell);
        wSol_      .emplace(nWell);
        dwSol_     .emplace(nWell);

        if constexpr (Backend::isGpu) {
            // CPU-side buffers for the well-solver round-trip.
            cpuWRes_ .resize(S_.getWW().N());
            cpuDwSol_.resize(S_.getWW().N());
        } else {
            // CPU scratch for well RHS (FlexibleSolver may modify RHS in-place).
            tmp_wRes_.emplace(nWell);
        }
    }
};

} // namespace Opm

#endif // OPM_SYSTEMPRECONDITIONER_HEADER_INCLUDED