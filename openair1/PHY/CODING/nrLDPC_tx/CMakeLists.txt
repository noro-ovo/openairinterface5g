# SPDX-License-Identifier: LicenseRef-CSSL-1.0

# Packed LDPC transmit chain (encoder + rate matching + bit interleaving to packed output), linked into libldpc
# together with nr_ldpc_tx_oai_encoder.c (LDPCencoder() for ldpctest). The ISA-specific kernels use function-level
# target attributes and are selected at run time, so the sources build for any -march baseline (x86-64-v2, armv8-a).
add_library(ldpc_tx OBJECT
  nr_ldpc_tx.c
  nr_ldpc_tx_scalar.c
  nr_ldpc_tx_avx2.c
  nr_ldpc_tx_avx512.c
  nr_ldpc_tx_neon.c
  nr_ldpc_tx_sve2.c
)
set_target_properties(ldpc_tx PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ldpc_tx PUBLIC log_headers)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
