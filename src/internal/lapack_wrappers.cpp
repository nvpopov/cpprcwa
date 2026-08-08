#include "lapack_wrappers.h"
#include <cpprcwa/errors.h>
#include <vector>

// Fortran LAPACK prototypes (ILP64-intel vs LP64-default handled by lapacke.h
// if available, but we declare directly to avoid that dependency).
extern "C" {
    // zgetrf: LU factorization
    void zgetrf_(const int* m, const int* n, cpprcwa::complex* a,
                 const int* lda, int* ipiv, int* info);
    // zgetrs: solve using LU
    void zgetrs_(const char* trans, const int* n, const int* nrhs,
                 const cpprcwa::complex* a, const int* lda, const int* ipiv,
                 cpprcwa::complex* b, const int* ldb, int* info);
    // zgetri: inverse from LU
    void zgetri_(const int* n, cpprcwa::complex* a, const int* lda,
                 const int* ipiv, cpprcwa::complex* work, const int* lwork,
                 int* info);
    // zgeev: eigendecomposition of general complex matrix
    void zgeev_(const char* jobvl, const char* jobvr, const int* n,
                cpprcwa::complex* a, const int* lda,
                cpprcwa::complex* w,
                cpprcwa::complex* vl, const int* ldvl,
                cpprcwa::complex* vr, const int* ldvr,
                cpprcwa::complex* work, const int* lwork,
                double* rwork, int* info);
}

namespace cpprcwa {
namespace internal {

void zgetri_inplace(int n, complex* A, int lda, int layer_for_error) {
    if (n == 0) return;
    // Reusable per-thread workspace: avoids the lwork query + fresh allocation
    // on every call (the S-matrix star product inverts many matrices).
    static thread_local std::vector<complex> work;
    static thread_local int work_size = 0;

    std::vector<int> ipiv(static_cast<size_t>(n));
    int info = 0;
    zgetrf_(&n, &n, A, &lda, ipiv.data(), &info);
    if (info != 0) {
        if (layer_for_error >= 0)
            throw error::SingularMatrixError("zgetrf", info, layer_for_error);
        throw error::LapackError("zgetrf", "zgetrf", info);
    }
    if (work_size < n) {               // grow (reused across calls of any size)
        int lwork = -1;
        complex wkopt;
        zgetri_(&n, A, &lda, ipiv.data(), &wkopt, &lwork, &info);
        work_size = static_cast<int>(wkopt.real());
        work.resize(static_cast<size_t>(work_size));
    }
    int lwork = work_size;
    zgetri_(&n, A, &lda, ipiv.data(), work.data(), &lwork, &info);
    if (info != 0)
        throw error::LapackError("zgetri", "zgetri", info);
}

ComplexMatrix zinverse(const ComplexMatrix& A, int layer_for_error) {
    ComplexMatrix B = A;  // copy
    zgetri_inplace(static_cast<int>(A.rows()), B.data(),
                   static_cast<int>(B.outerStride()), layer_for_error);
    return B;
}

std::vector<int> zgetrf_factor(int n, complex* A, int lda, int layer_for_error) {
    std::vector<int> ipiv(static_cast<size_t>(n));
    int info = 0;
    zgetrf_(&n, &n, A, &lda, ipiv.data(), &info);
    if (info != 0) {
        if (layer_for_error >= 0)
            throw error::SingularMatrixError("zgetrf", info, layer_for_error);
        throw error::LapackError("zgetrf", "zgetrf", info);
    }
    return ipiv;
}

void zgetrs_solve(int n, int nrhs, const complex* A, int lda, const int* ipiv,
                  complex* B, int ldb, int layer_for_error) {
    char trans = 'N';
    int info = 0;
    zgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    if (info != 0) {
        (void)layer_for_error;
        throw error::LapackError("zgetrs", "zgetrs", info);
    }
}

void zgesv(int n, int nrhs,
           complex* A, int lda,
           complex* B, int ldb,
           int layer_for_error) {
    std::vector<int> ipiv(n);
    int info = 0;
    char trans = 'N';
    zgetrf_(&n, &n, A, &lda, ipiv.data(), &info);
    if (info != 0) {
        if (layer_for_error >= 0)
            throw error::SingularMatrixError("zgetrf", info, layer_for_error);
        throw error::LapackError("zgesv", "zgetrf", info);
    }
    zgetrs_(&trans, &n, &nrhs, A, &lda, ipiv.data(), B, &ldb, &info);
    if (info != 0)
        throw error::LapackError("zgesv", "zgetrs", info);
}

void zgeev(int n,
           const complex* A, int lda,
           ComplexVector& w,
           complex* VR, int ldvr,
           int layer_for_error) {
    if (n == 0) { w.resize(0); return; }
    // Copy A (zgeev destroys it).
    std::vector<complex> a_copy(A, A + (size_t)lda * n);
    w.resize(n);
    std::vector<double> rwork(2 * n);
    char jobvl = 'N';
    char jobvr = (VR != nullptr) ? 'V' : 'N';
    complex vl_dummy;  // not referenced
    int info = 0;
    int lwork = -1;
    complex wkopt;
    zgeev_(&jobvl, &jobvr, &n, a_copy.data(), &lda,
           w.data(),
           &vl_dummy, &n,
           (VR != nullptr) ? VR : &vl_dummy,
           (VR != nullptr) ? &ldvr : &n,
           &wkopt, &lwork, rwork.data(), &info);
    if (info != 0) throw error::LapackError("zgeev(query)", "zgeev", info);
    lwork = static_cast<int>(wkopt.real());
    std::vector<complex> work(lwork);
    zgeev_(&jobvl, &jobvr, &n, a_copy.data(), &lda,
           w.data(),
           &vl_dummy, &n,
           (VR != nullptr) ? VR : &vl_dummy,
           (VR != nullptr) ? &ldvr : &n,
           work.data(), &lwork, rwork.data(), &info);
    if (info != 0) throw error::LapackError("zgeev", "zgeev", info);
    (void)layer_for_error;
}

} // namespace internal
} // namespace cpprcwa
