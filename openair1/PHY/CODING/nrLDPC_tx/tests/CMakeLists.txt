# SPDX-License-Identifier: LicenseRef-CSSL-1.0

# reference: the original encoder (ldpc_encoder.c) and the legacy rate matching / interleaving
add_executable(test_nr_ldpc_tx
  test_nr_ldpc_tx.cpp
  ../../nrLDPC_encoder/ldpc_encoder.c
  ../../nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.c
)
target_link_libraries(test_nr_ldpc_tx PRIVATE ldpc_tx GTest::gtest minimal_lib)
add_dependencies(tests test_nr_ldpc_tx)
add_test(NAME test_nr_ldpc_tx
  COMMAND ./test_nr_ldpc_tx)
