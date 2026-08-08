# Benchmarks (build with -DCPPRCWA_BUILD_BENCHMARKS=ON)
add_executable(bench_full_rt bench_full_rt.cpp)
target_link_libraries(bench_full_rt PRIVATE cpprcwa::cpprcwa)
