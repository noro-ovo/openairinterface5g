# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_sched_check test_sched_check.c ../sched_check.c)
target_include_directories(test_sched_check PRIVATE
  ${OPENAIR_DIR}
  ${OPENAIR_DIR}/common
  ${OPENAIR_DIR}/common/utils
  ${OPENAIR_DIR}/openair2/COMMON
  ${OPENAIR_DIR}/nfapi/open-nFAPI/nfapi/public_inc
  ${OPENAIR_DIR}/nfapi/open-nFAPI/common/public_inc)
target_compile_definitions(test_sched_check PRIVATE MAX_NUM_CCs=1)
add_dependencies(tests test_sched_check)
add_test(NAME test_sched_check COMMAND test_sched_check)

add_executable(test_sched_rules test_sched_rules.c ../sched_rules.c)
add_dependencies(tests test_sched_rules)
add_test(NAME test_sched_rules COMMAND test_sched_rules)

add_executable(test_sched_grid test_sched_grid.c ../sched_grid.c)
add_dependencies(tests test_sched_grid)
add_test(NAME test_sched_grid COMMAND test_sched_grid)

add_executable(test_sched_seed test_sched_seed.c ../sched_seed.c ../sched_grid.c)
target_include_directories(test_sched_seed PRIVATE
  ${OPENAIR_DIR}
  ${OPENAIR_DIR}/nfapi/open-nFAPI/nfapi/public_inc
  ${OPENAIR_DIR}/nfapi/open-nFAPI/common/public_inc)
target_compile_definitions(test_sched_seed PRIVATE MAX_NUM_CCs=1)
target_link_libraries(test_sched_seed PRIVATE nr_common minimal_lib)
add_dependencies(tests test_sched_seed)
add_test(NAME test_sched_seed COMMAND test_sched_seed)
