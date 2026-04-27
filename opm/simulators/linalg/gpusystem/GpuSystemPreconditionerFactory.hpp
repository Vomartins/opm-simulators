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
#include <opm/simulators/linalg/gpusystem/GpuSystemPreconditioner.hpp>

#include <opm/simulators/linalg/system/SystemTypes.hpp>
#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/is_gpu_operator.hpp>

#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>

#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuVector.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#endif

#include <functional>
#include <memory>

namespace Opm::gpusystem
{

// --------------------------------------------------------------------------
// GpuSystemSeqOpT
//
// GPU sequential system operator: wraps GpuSystemMatrixT for use with
// FlexibleSolver.  GpuSystemMatrixT provides mv/umv/usmv via cuSPARSE-backed
// SpMV, satisfying the AssembledLinearOperator interface required by
// Dune::MatrixAdapter.
// --------------------------------------------------------------------------
template<typename Scalar>
using GpuSystemSeqOpT = Dune::MatrixAdapter<
    GpuSystemMatrixT<Scalar>,
    GpuSystemVectorT<Scalar>,
    GpuSystemVectorT<Scalar>>;

} // namespace Opm::gpusystem

// --------------------------------------------------------------------------
// is_gpu_operator specialisation for GpuSystemSeqOpT
//
// GpuSystemSeqOpT::domain_type is GpuSystemVectorT, not GpuVector, so the
// primary is_gpu_operator template (which checks domain_type == GpuVector)
// would return false.  That causes FlexibleSolver_impl.hpp to attempt
// UMFPack<GpuSystemMatrixT>, which has no domain_type and yields a hard
// compile error.  This specialisation opts the type into the GPU operator
// code path, skipping the UMFPACK branch.
// --------------------------------------------------------------------------
namespace Opm {

template<typename Scalar>
struct is_gpu_operator<gpusystem::GpuSystemSeqOpT<Scalar>> {
    static constexpr bool value = true;
};

} // namespace Opm

namespace Opm::gpusystem {

// --------------------------------------------------------------------------
// detail::addGpuSystemCprSeq
//
// Registers the "gpu_system_cpr" creator in the PreconditionerFactory for
// GpuSystemSeqOpT<Scalar>.  Mirrors detail::addSystemCprSeq() in
// system/SystemPreconditionerFactory.hpp.
//
// The system-level weight calculator returns a GpuSystemVectorT; the
// GpuSystemPreconditioner expects a reservoir-only GpuVector, so the .res
// field is extracted in a wrapping lambda.
//
// cpuD (the CPU copy of the well-well block D) is retrieved via
// GpuSystemMatrixT::getCpuD(), where it is stored alongside the GPU block
// pointers by ISTLSolverGPUSystem::prepareGpuSystemSolver().
// --------------------------------------------------------------------------
namespace detail
{

template<typename Scalar>
void addGpuSystemCprSeq()
{
    using O      = GpuSystemSeqOpT<Scalar>;
    using F      = Opm::PreconditionerFactory<O, Dune::Amg::SequentialInformation>;
    using V      = GpuSystemVectorT<Scalar>;
    using GpuVec = gpuistl::GpuVector<Scalar>;
    using P      = Opm::PropertyTree;

    F::addCreator("gpu_system_cpr",
        [](const O& op, const P& prm,
           const std::function<V()>& sysWeightCalc,
           std::size_t pressureIndex) {
            // Wrap the system-level weights calc to return only the reservoir part.
            std::function<GpuVec()> resWeightCalc;
            if (sysWeightCalc) {
                resWeightCalc = [sysWeightCalc]() {
                    return sysWeightCalc().res;
                };
            }
            const auto& S = op.getmat();
            return std::make_shared<GpuSystemPreconditioner<Scalar>>(
                S, S.getCpuD(), resWeightCalc,
                static_cast<int>(pressureIndex), prm);
        });
}

} // namespace detail
} // namespace Opm::gpusystem

// --------------------------------------------------------------------------
// Full specialisations of Opm::StandardPreconditioners for the GPU system
// operator.  Must live in namespace Opm to match the primary template.
// Partial specialisations would be ambiguous with the is_gpu_operator_v guard
// in StandardPreconditioners_gpu_serial.hpp, so full specialisations are used,
// following the same pattern as system/SystemPreconditionerFactory.hpp.
// --------------------------------------------------------------------------
namespace Opm
{

template<class Operator, class Comm, typename>
struct StandardPreconditioners;

template<>
struct StandardPreconditioners<
    gpusystem::GpuSystemSeqOpT<double>,
    Dune::Amg::SequentialInformation,
    void>
{
    static void add() { gpusystem::detail::addGpuSystemCprSeq<double>(); }
};

template<>
struct StandardPreconditioners<
    gpusystem::GpuSystemSeqOpT<float>,
    Dune::Amg::SequentialInformation,
    void>
{
    static void add() { gpusystem::detail::addGpuSystemCprSeq<float>(); }
};

} // namespace Opm
