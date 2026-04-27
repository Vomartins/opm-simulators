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
#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuSparseMatrixGeneric.hpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuVector.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrixGeneric.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#endif

#include <dune/common/indices.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ostream>
#include <vector>

namespace Opm::gpusystem
{

// Reuse the dimension constants from the CPU side — do not duplicate.
using Opm::numResDofs;
using Opm::numWellDofs;

// --------------------------------------------------------------------------
// GPU matrix type aliases
//
// Square diagonal blocks (RR and WW) use GpuSparseMatrixWrapper which
// supports square BSR format with cuBLAS-backed block SpMV.
//
// Off-diagonal coupling blocks (RW and WR) have non-square block structure
// (numResDofs × numWellDofs = 3×4 and numWellDofs × numResDofs = 4×3
// respectively).  These are stored as scalar CSR matrices
// (GpuSparseMatrixGeneric with blockSize=1 and separate row/col counts)
// because the BSR GpuSparseMatrixWrapper only supports square blocks.
// --------------------------------------------------------------------------
template<typename Scalar>
using GpuRRMatrixT = gpuistl::GpuSparseMatrixWrapper<Scalar>;  // block = numResDofs × numResDofs (square)

template<typename Scalar>
using GpuRWMatrixT = gpuistl::GpuSparseMatrixGeneric<Scalar>;  // scalar CSR: nRes*numResDofs rows × nWell*numWellDofs cols

template<typename Scalar>
using GpuWRMatrixT = gpuistl::GpuSparseMatrixGeneric<Scalar>;  // scalar CSR: nWell*numWellDofs rows × nRes*numResDofs cols

template<typename Scalar>
using GpuWWMatrixT = gpuistl::GpuSparseMatrixWrapper<Scalar>;  // block = numWellDofs × numWellDofs (square)

// --------------------------------------------------------------------------
// GPU vector type aliases
//
// GpuVector<T> is flat (no block concept); block boundaries are implicit in
// the allocated size (nRows * blockDim scalars).
// --------------------------------------------------------------------------
template<typename Scalar>
using GpuResVectorT  = gpuistl::GpuVector<Scalar>;

template<typename Scalar>
using GpuWellVectorT = gpuistl::GpuVector<Scalar>;

// --------------------------------------------------------------------------
// GpuSystemVectorT
//
// GPU analogue of Dune::MultiTypeBlockVector<ResVectorT, WellVectorT>.
// Dune's heterogeneous container cannot be used on the device side, so we
// replace it with a plain aggregate of two flat GpuVectors.
//
// Memory layout mirrors the CPU convention:
//   res  — nResRows  * numResDofs  contiguous scalars
//   well — nWellRows * numWellDofs contiguous scalars
//
// fromCpu / toCpu provide the host↔device round-trips needed because the
// well solver remains on CPU (UMFPACK) per the GPU variant design.
// --------------------------------------------------------------------------
template<typename Scalar>
struct GpuSystemVectorT
{
    gpuistl::GpuVector<Scalar> res;   // reservoir unknowns
    gpuistl::GpuVector<Scalar> well;  // well unknowns

    // Scalar type exposed to Dune iterative solvers.
    using field_type = Scalar;

    // Default constructor — creates 0-element vectors.
    // Only valid for use as a placeholder; any BLAS op before resize will throw.
    GpuSystemVectorT() = default;

    // Size constructor — allocates nRes + nWell GPU scalar elements.
    // Use std::optional::emplace(nRes, nWell) to create properly-sized work vectors,
    // bypassing the non-resizing GpuVector copy-assignment.
    explicit GpuSystemVectorT(std::size_t nRes, std::size_t nWell)
        : res(nRes), well(nWell) {}

    // Upload from a CPU SystemVectorT (host→device, synchronous).
    static GpuSystemVectorT fromCpu(const SystemVectorT<Scalar>& v)
    {
        using namespace Dune::Indices;
        GpuSystemVectorT result(v[_0].dim(), v[_1].dim());
        result.res .copyFromHost(v[_0]);
        result.well.copyFromHost(v[_1]);
        return result;
    }

    // Download into an already-sized CPU SystemVectorT (device→host, synchronous).
    // The caller is responsible for ensuring v[_0] and v[_1] have the correct size.
    void toCpu(SystemVectorT<Scalar>& v) const
    {
        using namespace Dune::Indices;
        res .copyToHost(v[_0]);
        well.copyToHost(v[_1]);
    }

    // ------------------------------------------------------------------
    // BLAS operations — required by Dune Krylov iterative solvers.
    // Each operation delegates to GpuVector's cuBLAS-backed implementation.
    // ------------------------------------------------------------------

    // Zero-assign: v = 0  (used by solvers to initialise solution/residual).
    GpuSystemVectorT& operator=(Scalar s) {
        res  = s;
        well = s;
        return *this;
    }

    GpuSystemVectorT& operator+=(const GpuSystemVectorT& other) {
        res  += other.res;
        well += other.well;
        return *this;
    }

    GpuSystemVectorT& operator-=(const GpuSystemVectorT& other) {
        res  -= other.res;
        well -= other.well;
        return *this;
    }

    GpuSystemVectorT& operator*=(Scalar alpha) {
        res  *= alpha;
        well *= alpha;
        return *this;
    }

    // v += alpha * other
    void axpy(Scalar alpha, const GpuSystemVectorT& other) {
        res .axpy(alpha, other.res);
        well.axpy(alpha, other.well);
    }

    // Inner product: sum over res and well parts.
    Scalar dot(const GpuSystemVectorT& other) const {
        return res.dot(other.res) + well.dot(other.well);
    }

    // L2 norm: sqrt(dot(*this, *this)).
    Scalar two_norm() const {
        const Scalar d = dot(*this);
        return std::sqrt(d);
    }

    // Total number of scalar degrees of freedom.
    std::size_t dim() const {
        return res.dim() + well.dim();
    }
};

// Stream output for GpuSystemVectorT — prints summary info (no device-to-host copy).
// Required by Dune::RestartedFlexibleGMResSolver which streams w[i] on breakdown.
template<typename Scalar>
inline std::ostream& operator<<(std::ostream& os, const GpuSystemVectorT<Scalar>& v)
{
    os << "GpuSystemVectorT(res=" << v.res.dim() << ", well=" << v.well.dim() << ")";
    return os;
}

// --------------------------------------------------------------------------
// GpuSystemMatrixT
//
// GPU analogue of SystemMatrixT: a lightweight non-owning view over the four
// GPU sub-blocks of the 2×2 block system S = [[A,C],[B,D]].
//
// Unlike the CPU version, this class does NOT implement operator[](index_constant)
// row proxies — GpuSystemMatrixT is never passed to Dune::MatrixAdapter, so the
// indirection is unnecessary.  Sub-blocks are accessed via named getters.
//
// mv / umv / usmv delegate directly to GpuSparseMatrix's cuSPARSE-backed SpMV,
// preserving the same call semantics as the CPU SystemMatrixT.
// --------------------------------------------------------------------------
template<typename Scalar>
class GpuSystemMatrixT
{
public:
    // Dune linear-operator interface: 2 block-rows × 2 block-cols.
    using size_type  = std::size_t;
    using field_type = Scalar;
    static constexpr size_type N() { return 2; }
    static constexpr size_type M() { return 2; }

    // Sub-block pointers — set directly by the owning solver (ISTLSolverGPUSystem).
    const GpuRRMatrixT<Scalar>* A    = nullptr;  // (0,0) reservoir–reservoir  (GPU)
    const GpuRWMatrixT<Scalar>* C    = nullptr;  // (0,1) reservoir–well coupling (GPU)
    const GpuWRMatrixT<Scalar>* B    = nullptr;  // (1,0) well–reservoir coupling (GPU)
    const GpuWWMatrixT<Scalar>* D    = nullptr;  // (1,1) well–well (GPU)
    const WWMatrixT<Scalar>*    cpuD = nullptr;  // (1,1) well–well CPU copy, needed
                                                 //       by the CPU well sub-solver

    // Named sub-block accessors (replaces row-proxy operator[]).
    const GpuRRMatrixT<Scalar>& getRR()  const { return *A; }
    const GpuRWMatrixT<Scalar>& getRW()  const { return *C; }
    const GpuWRMatrixT<Scalar>& getWR()  const { return *B; }
    const GpuWWMatrixT<Scalar>& getWW()  const { return *D; }
    const WWMatrixT<Scalar>&    getCpuD() const { return *cpuD; }

    // y = S * x  (overwrites y)
    void mv(const GpuSystemVectorT<Scalar>& x, GpuSystemVectorT<Scalar>& y) const
    {
        A->mv (x.res,  y.res);   C->umv(x.well, y.res);
        B->mv (x.res,  y.well);  D->umv(x.well, y.well);
    }

    // y += S * x
    void umv(const GpuSystemVectorT<Scalar>& x, GpuSystemVectorT<Scalar>& y) const
    {
        A->umv(x.res,  y.res);   C->umv(x.well, y.res);
        B->umv(x.res,  y.well);  D->umv(x.well, y.well);
    }

    // y += alpha * S * x
    void usmv(Scalar alpha, const GpuSystemVectorT<Scalar>& x, GpuSystemVectorT<Scalar>& y) const
    {
        A->usmv(alpha, x.res,  y.res);   C->usmv(alpha, x.well, y.res);
        B->usmv(alpha, x.res,  y.well);  D->usmv(alpha, x.well, y.well);
    }
};

// --------------------------------------------------------------------------
// makeNonSquareScalarGpuMatrix
//
// Converts a DUNE BCRSMatrix<FieldMatrix<S, R, C>> with non-square blocks
// (R != C) into a scalar CSR GpuSparseMatrixGeneric<S> suitable for use as
// the off-diagonal coupling blocks B (WR) and C (RW) in GpuSystemMatrixT.
//
// Layout: each block entry (blockRow i, blockCol j) with value M expands to
// R×C scalar entries at scalar rows [i*R .. i*R+R-1] and scalar columns
// [j*C .. j*C+C-1].  The resulting matrix has:
//   rows = nBlockRows * R,   cols = nBlockCols * C,   nnz = nBlockNNZ * R * C
//
// Column indices are written in C-major (row-minor) order within each block
// so that a single scalar row corresponds to a single block row r.
// --------------------------------------------------------------------------
template<typename Scalar, int R, int C>
gpuistl::GpuSparseMatrixGeneric<Scalar>
makeNonSquareScalarGpuMatrix(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, R, C>>& blockMat)
{
    const int nBlockRows = static_cast<int>(blockMat.N());
    const int nBlockCols = static_cast<int>(blockMat.M());
    const int nRows      = nBlockRows * R;
    const int nCols      = nBlockCols * C;
    const int nnz        = static_cast<int>(blockMat.nonzeroes()) * R * C;

    std::vector<int>    rowPtrs(nRows + 1, 0);
    std::vector<int>    colIndices(nnz);
    std::vector<Scalar> values(nnz);

    // Build row pointers: each scalar row i*R+r contains nnz_in_block_row * C entries.
    for (int bi = 0; bi < nBlockRows; ++bi) {
        const int nnz_in_row = static_cast<int>(
            std::distance(blockMat[bi].begin(), blockMat[bi].end()));
        for (int r = 0; r < R; ++r) {
            rowPtrs[bi * R + r + 1] = rowPtrs[bi * R + r] + nnz_in_row * C;
        }
    }

    // Fill column indices and values.
    int idx = 0;
    for (int bi = 0; bi < nBlockRows; ++bi) {
        for (int r = 0; r < R; ++r) {
            for (auto it = blockMat[bi].begin(); it != blockMat[bi].end(); ++it) {
                const int  bj    = static_cast<int>(it.index());
                const auto& blk  = *it;
                for (int c = 0; c < C; ++c) {
                    colIndices[idx] = bj * C + c;
                    values[idx]     = blk[r][c];
                    ++idx;
                }
            }
        }
    }

    return gpuistl::GpuSparseMatrixGeneric<Scalar>(
        values.data(),
        rowPtrs.data(),
        colIndices.data(),
        static_cast<std::size_t>(nnz),
        1,                                   // blockSize = 1 (scalar CSR)
        static_cast<std::size_t>(nRows),
        static_cast<std::size_t>(nCols));
}

} // namespace Opm::gpusystem
