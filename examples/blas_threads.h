// OpenBLAS runtime thread selection for the examples.
//
// Many RCWA matrices (2nG×2nG) are small and saturate at a modest thread
// count; using all logical cores (including Hyper-Threading siblings) adds
// thread-pool overhead. We therefore cap at the number of *physical* cores.
// Overridable via the OPENBLAS_NUM_THREADS environment variable or a
// --threads command-line argument.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <utility>

namespace blas_threads {

// Number of physical cores on Linux: count unique (physical id, core id)
// pairs in /proc/cpuinfo (excludes HT siblings). Falls back to the logical
// core count on non-Linux or if parsing fails.
inline int physical_cores() {
    std::FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return std::max(1u, std::thread::hardware_concurrency());
    char line[256];
    int phys = -1, core = -1;
    std::set<std::pair<int, int>> pairs;
    while (std::fgets(line, sizeof(line), f)) {
        int v;
        if (std::sscanf(line, "physical id : %d", &v) == 1) {
            phys = v;
        } else if (std::sscanf(line, "core id : %d", &v) == 1) {
            core = v;
            if (phys >= 0) pairs.insert({phys, core});
        }
    }
    std::fclose(f);
    if (!pairs.empty()) return (int)pairs.size();
    return std::max(1u, std::thread::hardware_concurrency());
}

// Pick the BLAS thread count, capped at the physical core count.
inline int choose(int ncores) {
    const char* env = std::getenv("OPENBLAS_NUM_THREADS");
    if (env && *env) return std::atoi(env);
    return std::clamp(ncores, 1, physical_cores());
}

} // namespace blas_threads
