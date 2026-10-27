# SPDX-License-Identifier: LicenseRef-CSSL-1.0

# rate matching and decoder side; the transport-block encoder is chosen per library (nrLDPC_coding_segment_encoder.c in
# ldpc_orig, nrLDPC_coding_segment_encoder_packed.c in ldpc, see PHY/CODING/CMakeLists.txt)
add_library(ldpc_segment OBJECT
  nr_rate_matching.c
  nrLDPC_coding_segment_decoder.c
  nrLDPC_llr_prescale.c
)
target_link_libraries(ldpc_segment PUBLIC log_headers)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
