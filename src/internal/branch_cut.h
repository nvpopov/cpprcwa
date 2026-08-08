#pragma once

#include <cpprcwa/types.h>

namespace cpprcwa {

// Branch-cut helper (§6.6): force Im(q) >= 0 (evanescent waves decay in +z).
ComplexVector apply_branch_cut(const ComplexVector& q);

} // namespace cpprcwa
