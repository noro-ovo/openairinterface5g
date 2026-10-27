# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_nr_polar test_nr_polar.cpp nr_polar_test_vectors.c)
target_link_libraries(test_nr_polar PRIVATE GTest::gtest polar crc_byte minimal_lib)
add_dependencies(tests test_nr_polar)
add_test(NAME test_nr_polar
  COMMAND ./test_nr_polar)
