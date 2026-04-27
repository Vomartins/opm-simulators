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

#include <config.h>

#include <opm/simulators/linalg/FlexibleSolver_impl.hpp>
#include <opm/simulators/linalg/PreconditionerFactory_impl.hpp>
#include <opm/simulators/linalg/gpusystem/GpuSystemPreconditioner.hpp>
#include <opm/simulators/linalg/gpusystem/GpuSystemPreconditionerFactory.hpp>

// Explicit instantiations for GpuSystemPreconditioner and the CPU well
// sub-solver (Dune::FlexibleSolver over the well-only operator).
//
// GpuSystemPreconditioner is header-only (all methods inline), so only
// the class itself and the CPU sub-solver need explicit instantiation here.
//
// FlexibleSolverWrapper for the reservoir (GPU) sub-solvers and the outer
// Dune::FlexibleSolver<GpuSystemSeqOpT<T>> are instantiated in
// FlexibleSolver_gpu_instantiate.cpp (a CUDA/HIP translation unit).
// They cannot be instantiated here because FlexibleSolver_impl.hpp pulls
// in umfpack.hh, which requires domain_type from the matrix type —
// GpuSystemMatrixT does not satisfy that requirement.

#define INSTANTIATE_GPU_SYSTEM(T)                                               \
    /* GpuSystemPreconditioner (3-stage GPU+CPU preconditioner) */              \
    template class Opm::gpusystem::GpuSystemPreconditioner<T>;                  \
    /* CPU well sub-solver: FlexibleSolver over the D (well×well) block */      \
    template class Dune::FlexibleSolver<                                         \
        Dune::MatrixAdapter<Opm::WWMatrixT<T>,                                  \
                            Opm::WellVectorT<T>,                                \
                            Opm::WellVectorT<T>>>;

INSTANTIATE_GPU_SYSTEM(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_GPU_SYSTEM(float)
#endif
