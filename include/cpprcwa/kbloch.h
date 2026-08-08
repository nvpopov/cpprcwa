#pragma once

#include <utility>
#include <Eigen/Dense>
#include <cpprcwa/types.h>

namespace cpprcwa {

// Compute reciprocal lattice vectors.
//   Lk1 = [ L2[1]/d, -L2[0]/d ]
//   Lk2 = [-L1[1]/d,  L1[0]/d ]
// where d = L1 x L2 (scalar 2D cross product).
std::pair<Eigen::Vector2d, Eigen::Vector2d>
Lattice_Reciprocate(const Eigen::Vector2d& L1, const Eigen::Vector2d& L2);

// Select the set of reciprocal-lattice G vectors (integer pairs) truncated
// to approximately nG points.
//   method=0: circular selection (default, preserves symmetry)
//   method=1: parallelogramic selection
// Returns (G, nG_out) where G is (nG_out, 2) integer matrix and nG_out <= nG
// (may be smaller due to degeneracy handling in the circular mode).
std::pair<IntMatrix, int>
Lattice_getG(int nG, const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2,
             int method = 0);

// Compute in-plane k-vectors for each G point: kx[i], ky[i].
// kx0, ky0 are complex because omega is complex (§6.1).
void Lattice_SetKs(const IntMatrix& G,
                   complex kx0, complex ky0,
                   const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2,
                   ComplexVector& kx, ComplexVector& ky);

} // namespace cpprcwa
