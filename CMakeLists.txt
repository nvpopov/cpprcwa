cmake_minimum_required(VERSION 3.18)
project(cpprcwa LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CPPRCWA_BUILD_TESTS    "Build cpprcwa unit tests"    ON)
option(CPPRCWA_BUILD_EXAMPLES "Build cpprcwa examples"      ON)
option(CPPRCWA_BUILD_BENCHMARKS "Build cpprcwa benchmarks"  OFF)
option(CPPRCWA_USE_CUDA      "Enable CUDA GPU backend"      OFF)

# ── Release optimization ────────────────────────────────────────────────────
# Explicit, tuned compiler flags on top of CMake's -O3. All are scoped to the
# Release configuration and are safe for the validated FP results (branch
# cuts, evanescent fields).
option(CPPRCWA_USE_NATIVE   "Use -march=native / -mtune=native (CPU-specific SIMD)" ON)
option(CPPRCWA_USE_LTO      "Enable link-time optimization (-flto)"               ON)
option(CPPRCWA_USE_FASTMATH "Enable -ffast-math (may alter FP rounding; opt-in)"  OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(flag -O3 -funroll-loops -fno-math-errno -ffp-contract=fast)
        add_compile_options($<$<CONFIG:Release>:${flag}>)
    endforeach()
    if(CPPRCWA_USE_NATIVE)
        add_compile_options($<$<CONFIG:Release>:-march=native>)
        add_compile_options($<$<CONFIG:Release>:-mtune=native>)
    endif()
    if(CPPRCWA_USE_LTO)
        add_compile_options($<$<CONFIG:Release>:-flto>)
        add_link_options($<$<CONFIG:Release>:-flto>)
    endif()
    if(CPPRCWA_USE_FASTMATH)
        add_compile_options($<$<CONFIG:Release>:-ffast-math>)
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ── Dependencies ───────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    # Fallback for older Eigen find modules
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_package(FFTW3 REQUIRED)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# ── Core library ───────────────────────────────────────────────────────────
add_library(cpprcwa STATIC
    src/kbloch.cpp
    src/fft_funs.cpp
    src/internal/utils.cpp
    src/internal/branch_cut.cpp
    src/internal/lapack_wrappers.cpp
    src/rcwa.cpp
)
add_library(cpprcwa::cpprcwa ALIAS cpprcwa)

target_include_directories(cpprcwa
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cpprcwa
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        FFTW3::fftw3
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
)

target_compile_features(cpprcwa PUBLIC cxx_std_17)
# Use BLAS (zgemm) for Eigen matrix products instead of Eigen's internal
# single-threaded kernels — a large win for the S-matrix star product.
target_compile_definitions(cpprcwa PRIVATE EIGEN_USE_BLAS)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(cpprcwa PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── Subdirectories ─────────────────────────────────────────────────────────
if(CPPRCWA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(CPPRCWA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(CPPRCWA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Install ────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS cpprcwa
    EXPORT cpprcwaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/cpprcwa
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
