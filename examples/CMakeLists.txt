add_executable(ex1_square_lattice ex1_square_lattice.cpp)
target_link_libraries(ex1_square_lattice PRIVATE cpprcwa::cpprcwa)

add_executable(ex2_two_layers ex2_two_layers.cpp)
target_link_libraries(ex2_two_layers PRIVATE cpprcwa::cpprcwa)

add_executable(ex4_hexagonal ex4_hexagonal.cpp)
target_link_libraries(ex4_hexagonal PRIVATE cpprcwa::cpprcwa)

add_executable(ex_euv_multilayer ex_euv_multilayer.cpp)
target_link_libraries(ex_euv_multilayer PRIVATE cpprcwa::cpprcwa)

add_executable(ex_euv_absorber ex_euv_absorber.cpp)
target_link_libraries(ex_euv_absorber PRIVATE cpprcwa::cpprcwa)

add_executable(ex_quasi1d_absorber ex_quasi1d_absorber.cpp)
target_link_libraries(ex_quasi1d_absorber PRIVATE cpprcwa::cpprcwa)
