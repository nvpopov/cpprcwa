#include "branch_cut.h"

namespace cpprcwa {

ComplexVector apply_branch_cut(const ComplexVector& q) {
    ComplexVector result = q;
    for (int i = 0; i < q.size(); ++i) {
        if (result(i).imag() < 0.0) result(i) = -result(i);
    }
    return result;
}

} // namespace cpprcwa
