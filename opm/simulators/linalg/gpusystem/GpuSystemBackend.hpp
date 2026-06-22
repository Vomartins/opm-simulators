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

#include <dune/common/parallel/communication.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <functional>

namespace Opm::gpusystem
{

/// Backend traits for the GPU instantiation of SystemPreconditioner.
///
/// Bundles all GPU-specific type aliases and static helpers so that
/// the unified SystemPreconditioner<Backend> can be instantiated for
/// GPU without a separate class.
template <typename ScalarT>
struct GpuSystemBackend
{
    static constexpr bool isGpu      = true;
    static constexpr bool isParallel = false;
    static constexpr bool hasPerfectUpdate = false;

    using ScalarType   = ScalarT;
    using SysMatrix    = GpuSystemMatrixT<ScalarT>;
    using SysVector    = GpuSystemVectorT<ScalarT>;
    using GpuVec       = gpuistl::GpuVector<ScalarT>;
    using ResWork      = GpuVec;
    using WellWork     = GpuVec;

    // Unused on GPU — FlexibleSolverWrapper creates its own operator.
    // Defined so that unique_ptr<ResOperator> compiles (always null).
    struct ResOperator {};

    using ResCommType  = Dune::Amg::SequentialInformation;

    using SerialComm   = Dune::Communication<int>;
    using GpuResMatrix = GpuRRMatrixT<ScalarT>;
    using ResSolver    = gpuistl::detail::FlexibleSolverWrapper<GpuResMatrix, GpuVec, SerialComm>;

    using WellOperator = Dune::MatrixAdapter<WWMatrixT<ScalarT>,
                                             WellVectorT<ScalarT>,
                                             WellVectorT<ScalarT>>;
    using WellSolver   = Dune::FlexibleSolver<WellOperator>;
    using WeightsCalc  = std::function<GpuVec()>;

    // --- System vector access ---
    static GpuVec&       getRes(SysVector& v)        { return v.res; }
    static const GpuVec& getRes(const SysVector& v)  { return v.res; }
    static GpuVec&       getWell(SysVector& v)       { return v.well; }
    static const GpuVec& getWell(const SysVector& v) { return v.well; }

    // --- Work-vector sizing ---
    // Returns the flat scalar count (GPU GpuVector constructor argument).
    static std::size_t resWorkSize(const SysMatrix& S)
    { return S.getRR().N() * numResDofs; }

    static std::size_t wellWorkSize(const SysMatrix& S)
    { return S.getWW().N() * numWellDofs; }

    // --- CPU well-well matrix for the well sub-solver ---
    // GpuSystemMatrixT stores a separate CPU copy of D alongside the GPU blocks.
    static const WWMatrixT<ScalarT>& getWellMatrixCpu(const SysMatrix& S)
    { return S.getCpuD(); }

    // --- Reservoir sub-solver update ---
    // FlexibleSolverWrapper exposes a direct update() method.
    static void updateResSolver(ResSolver& s) { s.update(); }
};

} // namespace Opm::gpusystem
