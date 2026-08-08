# Catch2 v3 is installed system-wide (find_package).
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    # FetchContent fallback for systems without Catch2 installed.
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0)
    FetchContent_MakeAvailable(Catch2)
    find_package(Catch2 3 REQUIRED)
endif()

add_executable(cpprcwa_tests
    test_main.cpp
    test_kbloch.cpp
    test_fft_funs.cpp
    test_rcwa.cpp
)
target_compile_definitions(cpprcwa_tests PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)
target_link_libraries(cpprcwa_tests PRIVATE cpprcwa::cpprcwa Catch2::Catch2WithMain)
target_compile_features(cpprcwa_tests PRIVATE cxx_std_17)

include(CTest)
include(Catch)
catch_discover_tests(cpprcwa_tests)
