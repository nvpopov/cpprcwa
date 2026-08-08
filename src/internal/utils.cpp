#include "utils.h"

namespace cpprcwa {
namespace utils {

ComplexVector diag(const ComplexMatrix& A) {
    int n = std::min(A.rows(), A.cols());
    ComplexVector d(n);
    for (int i = 0; i < n; ++i) d(i) = A(i, i);
    return d;
}

ComplexMatrix eye(int n) {
    return ComplexMatrix::Identity(n, n);
}

ComplexMatrix makeDiag(const ComplexVector& v) {
    return v.asDiagonal().toDenseMatrix();
}

ComplexMatrix blockDiag(const ComplexMatrix& A, const ComplexMatrix& B) {
    ComplexMatrix M = ComplexMatrix::Zero(A.rows() + B.rows(), A.cols() + B.cols());
    M.topLeftCorner(A.rows(), A.cols())     = A;
    M.bottomRightCorner(B.rows(), B.cols()) = B;
    return M;
}

ComplexMatrix vstack(const ComplexMatrix& A, const ComplexMatrix& B) {
    assert(A.cols() == B.cols());
    ComplexMatrix M(A.rows() + B.rows(), A.cols());
    M.topRows(A.rows())    = A;
    M.bottomRows(B.rows()) = B;
    return M;
}

ComplexMatrix hstack(const ComplexMatrix& A, const ComplexMatrix& B) {
    assert(A.rows() == B.rows());
    ComplexMatrix M(A.rows(), A.cols() + B.cols());
    M.leftCols(A.cols())   = A;
    M.rightCols(B.cols())  = B;
    return M;
}

} // namespace utils
} // namespace cpprcwa
