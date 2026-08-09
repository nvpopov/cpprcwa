# Benchmarks (build with -DCPPRCWA_BUILD_BENCHMARKS=ON)
add_executable(bench_full_rt bench_full_rt.cpp)
target_link_libraries(bench_full_rt PRIVATE cpprcwa::cpprcwa)

# GPU-vs-CPU zgeev feasibility spike (PLAN.md Phase 9). CPU-only target; the
# GPU variant is compiled manually with nvcc -DUSE_CUDA -lcusolver (see the
# header comment in bench_zgeev.cpp).
add_executable(bench_zgeev bench_zgeev.cpp)
target_link_libraries(bench_zgeev PRIVATE cpprcwa::cpprcwa)
