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

#include <opm/simulators/linalg/system/SystemTypes.hpp>
#include <opm/simulators/linalg/FlexibleSolver.hpp>

#include <dune/common/indices.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <functional>
#include <memory>
#include <type_traits>

namespace Opm
{

/// Backend traits for the CPU instantiation of SystemPreconditioner.
///
/// Bundles all type aliases and static helpers that differ between the
/// CPU and GPU paths.  The SystemPreconditioner is templated on a
/// Backend that satisfies this interface.
template <typename ScalarT, class ResOp, class ResComm = Dune::Amg::SequentialInformation>
struct CpuSystemBackend
{
    static constexpr bool isGpu      = false;
    static constexpr bool isParallel = !std::is_same_v<ResComm, Dune::Amg::SequentialInformation>;
    static constexpr bool hasPerfectUpdate = true;

    using ScalarType   = ScalarT;
    using SysMatrix    = SystemMatrixT<ScalarT>;
    using SysVector    = SystemVectorT<ScalarT>;
    using ResWork      = ResVectorT<ScalarT>;
    using WellWork     = WellVectorT<ScalarT>;
    using ResOperator  = ResOp;
    using ResCommType  = ResComm;
    using ResSolver    = Dune::FlexibleSolver<ResOp>;
    using WellOperator = Dune::MatrixAdapter<WWMatrixT<ScalarT>,
                                             WellVectorT<ScalarT>,
                                             WellVectorT<ScalarT>>;
    using WellSolver   = Dune::FlexibleSolver<WellOperator>;
    using WeightsCalc  = std::function<ResVectorT<ScalarT>()>;

    // --- System vector access ---
    static ResVectorT<ScalarT>&        getRes(SysVector& v)        { return v[Dune::Indices::_0]; }
    static const ResVectorT<ScalarT>&  getRes(const SysVector& v)  { return v[Dune::Indices::_0]; }
    static WellVectorT<ScalarT>&       getWell(SysVector& v)       { return v[Dune::Indices::_1]; }
    static const WellVectorT<ScalarT>& getWell(const SysVector& v) { return v[Dune::Indices::_1]; }

    // --- Work-vector sizing ---
    // Returns the number of blocks (CPU BlockVector constructor argument).
    static std::size_t resWorkSize(const SysMatrix& S)  { return S.getRR().N(); }
    static std::size_t wellWorkSize(const SysMatrix& S)  { return S.getWW().N(); }

    // --- CPU well-well matrix for the well sub-solver ---
    static const WWMatrixT<ScalarT>& getWellMatrixCpu(const SysMatrix& S)
    { return S.getWW(); }

    // --- Reservoir sub-solver update ---
    // FlexibleSolver has no direct update(); must go through preconditioner().
    static void updateResSolver(ResSolver& s) { s.preconditioner().update(); }
};

} // namespace Opm
