# FindFFTW3.cmake — locate FFTW3 double-precision library.
#
# Defines:
#   FFTW3_FOUND        - TRUE if fftw3 found
#   FFTW3_INCLUDE_DIR  - include directory containing fftw3.h
#   FFTW3_LIBRARY      - the fftw3 library to link against
#
# Imported target:
#   FFTW3::fftw3
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR)

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_LIBRARY}")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
