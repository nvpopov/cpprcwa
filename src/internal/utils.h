#pragma once
#include <cpprcwa/types.h>

namespace cpprcwa {
namespace utils {

// Extract diagonal of an n×n matrix as a length-n vector.
ComplexVector diag(const ComplexMatrix& A);

// n×n identity.
ComplexMatrix eye(int n);

// Form a diagonal matrix from a vector.
ComplexMatrix makeDiag(const ComplexVector& v);

// Block-diagonal stacking of two matrices: [[A, 0], [0, B]].
ComplexMatrix blockDiag(const ComplexMatrix& A, const ComplexMatrix& B);

// Vertical / horizontal stack.
ComplexMatrix vstack(const ComplexMatrix& A, const ComplexMatrix& B);
ComplexMatrix hstack(const ComplexMatrix& A, const ComplexMatrix& B);

} // namespace utils
} // namespace cpprcwa
