# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_library(coding MODULE
            3gpplte_sse.c
            3gpplte.c
            3gpplte_turbo_decoder_sse_8bit.c
            3gpplte_turbo_decoder_sse_16bit.c
            3gpplte_turbo_decoder_avx2_16bit.c
            3gpplte_turbo_decoder.c
)
set_target_properties(coding PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(coding PRIVATE log_headers)

set(NR_LDPC_SELECT nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_select.c)

# OAI's original decoder and encoder (reference of ldpctest)
add_library(ldpc_orig MODULE
            nrLDPC_decoder/nrLDPC_decoder.c
            nrLDPC_encoder/ldpc_encoder.c
            ${NR_LDPC_SELECT}
)
set_target_properties(ldpc_orig PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc_orig PRIVATE ldpc_segment)

# libldpc_oaidec.so: segment coding library with OAI's decoder only (= libldpc.so before the fast decoder);
# select it with --loader.ldpc.shlibversion _oaidec (softmodems, nr_ulsim, nr_dlsim) or ldpctest -v _oaidec
add_library(ldpc_oaidec MODULE
            nrLDPC_decoder/nrLDPC_decoder.c
            nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
            ${NR_LDPC_SELECT}
)
set_target_properties(ldpc_oaidec PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc_oaidec PRIVATE ldpc_segment)

# libldpc.so: segment coding library with the fast decoder (nrLDPC_decoder_fast/) and OAI's decoder, chosen at run time
# by nrLDPC_coding_segment.decoder = fast|oai (config file section or command line) or env OAI_LDPC_DECODER, default
# NR_LDPC_DECODER_DEFAULT (see nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_select.h)
add_boolean_option(LDPC_DECODER_FAST ON "Build the fast LDPC decoder into libldpc.so (OFF: libldpc.so has OAI's decoder only)" OFF)
set(NR_LDPC_DECODER_DEFAULT "fast" CACHE STRING "LDPC decoder of libldpc.so unless selected at run time: fast or oai")
set_property(CACHE NR_LDPC_DECODER_DEFAULT PROPERTY STRINGS fast oai)
if (LDPC_DECODER_FAST)
  add_subdirectory(nrLDPC_decoder_fast)
  add_library(ldpc MODULE
              nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
              ${NR_LDPC_SELECT}
              nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_decoder_oai_renamed.c
              $<TARGET_OBJECTS:ldpc_decoder_fast>
  )
  target_compile_definitions(ldpc PRIVATE NR_LDPC_HAVE_FAST_DECODER NR_LDPC_DECODER_DEFAULT="${NR_LDPC_DECODER_DEFAULT}")
  target_include_directories(ldpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nrLDPC_decoder_fast/upstream/include)
else()
  add_library(ldpc MODULE
              nrLDPC_decoder/nrLDPC_decoder.c
              nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
              ${NR_LDPC_SELECT}
  )
endif()
set_target_properties(ldpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc PRIVATE ldpc_segment)

add_dependencies(ldpctest ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr-softmodem ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr-uesoftmodem ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr_ulsim ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr_ulschsim ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr_dlsim ldpc ldpc_orig ldpc_oaidec)
add_dependencies(nr_dlschsim ldpc ldpc_orig ldpc_oaidec)
if (ENABLE_TESTS)
  add_dependencies(tests ldpc ldpc_orig ldpc_oaidec)
endif()

add_library(crc_byte OBJECT crc_byte.c crc_fold.c)

add_subdirectory(nrLDPC_coding)
add_subdirectory(nrPolar_tools)
add_subdirectory(nrSmallBlock)
