#pragma once

#include <cpprcwa/types.h>

namespace cpprcwa {
namespace internal {

// Thin wrappers around LAPACK routines used by rcwa.cpp.
// Each throws cpprcwa::error::LapackError on info != 0 (or SingularMatrixError
// for factorization failures).

// Solve A * X = B for general complex matrix (uses zgetrf + zgetrs).
// Overwrites A with its LU factorization, overwrites B with X.
void zgesv(int n, int nrhs,
           complex* A, int lda,
           complex* B, int ldb,
           int layer_for_error = -1);

// Invert an n×n complex matrix in place. Uses zgetrf + zgetri.
// workspace is allocated internally.
void zgetri_inplace(int n, complex* A, int lda, int layer_for_error = -1);

// Compute inverse, returning a new matrix. Convenience wrapper.
ComplexMatrix zinverse(const ComplexMatrix& A, int layer_for_error = -1);

// LU-factorize an n×n complex matrix in place (zgetrf). Returns the pivot
// vector for use with zgetrs_solve. Throws SingularMatrixError on failure.
std::vector<int> zgetrf_factor(int n, complex* A, int lda, int layer_for_error = -1);

// Solve A·X = B using a factorization produced by zgetrf_factor. A holds the
// LU factors, ipiv the pivots. B (n×nrhs, column-major, ldb) is overwritten
// with X. O(n²·nrhs) — much cheaper than forming the inverse.
void zgetrs_solve(int n, int nrhs, const complex* A, int lda, const int* ipiv,
                  complex* B, int ldb, int layer_for_error = -1);

// Eigendecomposition of a general (non-Hermitian) complex matrix.
// Computes eigenvalues (w), right eigenvectors (VR).
// A is not modified.
void zgeev(int n,
           const complex* A, int lda,
           ComplexVector& w,
           complex* VR, int ldvr,
           int layer_for_error = -1);

} // namespace internal
} // namespace cpprcwa
