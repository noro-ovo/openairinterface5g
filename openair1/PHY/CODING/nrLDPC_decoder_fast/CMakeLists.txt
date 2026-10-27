# SPDX-License-Identifier: LicenseRef-CSSL-1.0
#
# Fast NR LDPC decoder (layered min-sum, int8, run-time dispatched NEON / SVE2 / AVX2 / AVX-512 / portable C),
# vendored in upstream/ (refresh with sync_from_upstream.sh, see README.md).
#
# ldpc_decoder_fast: object library linked into libldpc.so (see ../CMakeLists.txt). Its OAI entry points are
# renamed to nrLDPC_fast_LDPCdecoder / _LDPCinit / _LDPCshutdown; the segment library's decoder selection
# (nrLDPC_coding_segment/nrLDPC_decoder_select.c) calls them. All its symbols are hidden (-fvisibility=hidden), so
# libldpc.so exports exactly what it exported before (LDPCdecoder & co. are the selector's).
#
# The source list is a glob: a new upstream version that adds or removes kernel files needs no edit here. Every
# kernel file compiles on every architecture (the ISA-specific ones are empty elsewhere); the x86 kernels carry
# `#pragma GCC target("avx2")` / ("avx512f,avx512bw,avx512vl,avx2") and the aarch64 SVE2 kernel
# `#pragma GCC target("arch=armv8.5-a+sve2")`, selected at run time (cpuid / getauxval(AT_HWCAP2) & HWCAP2_SVE2), so
# OAI's global -march (x86-64 native/skylake-avx512, aarch64 armv8.2-a cross or -mcpu/-march=native) only sets the
# baseline. LDPC_DEC_IMPL=scalar|avx2|avx512|neon|sve2|ref forces an implementation (tests / measurements).

file(GLOB NR_LDPC_DEC_FAST_SRC CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/upstream/src/*.c
     ${CMAKE_CURRENT_SOURCE_DIR}/upstream/tables/*.c)

add_library(ldpc_decoder_fast OBJECT ${NR_LDPC_DEC_FAST_SRC})
target_include_directories(ldpc_decoder_fast PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/upstream/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/upstream/src
                           ${OPENAIR_DIR})
target_compile_definitions(ldpc_decoder_fast PRIVATE
                           LDPC_DEC_OAI_HEADERS        # OAI's own t_nrLDPC_dec_params / decode_abort_t
                           LDPC_DEC_OAI_HAS_QM         # AGC target per modulation order from p_decParams->Qm
                           LDPCdecoder=nrLDPC_fast_LDPCdecoder
                           LDPCinit=nrLDPC_fast_LDPCinit
                           LDPCshutdown=nrLDPC_fast_LDPCshutdown)
# -O3 as tuned upstream (RelWithDebInfo would give -O2); the vendored code keeps its own warning set
target_compile_options(ldpc_decoder_fast PRIVATE -fvisibility=hidden $<$<NOT:$<CONFIG:Debug>>:-O3> -Wno-unused-function)
target_link_libraries(ldpc_decoder_fast PRIVATE log_headers)

# unit test of the vendored decoder (bit-exactness of every available kernel against the scalar reference, spec
# encoder + syndrome for all Z, AGC/CRC-mode paths): `ninja test_ldpc_decoder_fast && ./test_ldpc_decoder_fast --quick`
add_executable(test_ldpc_decoder_fast EXCLUDE_FROM_ALL
               ${CMAKE_CURRENT_SOURCE_DIR}/upstream/test/test_dec.c
               ${CMAKE_CURRENT_SOURCE_DIR}/upstream/test/tutil.c
               $<TARGET_OBJECTS:ldpc_decoder_fast>)
target_include_directories(test_ldpc_decoder_fast PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/upstream/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/upstream/src
                           ${CMAKE_CURRENT_SOURCE_DIR}/upstream/test)
target_compile_options(test_ldpc_decoder_fast PRIVATE -Wno-unused-function)
set_target_properties(test_ldpc_decoder_fast PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(test_ldpc_decoder_fast PRIVATE m pthread)
if (ENABLE_TESTS)
  add_dependencies(tests test_ldpc_decoder_fast)
  add_test(NAME test_ldpc_decoder_fast COMMAND test_ldpc_decoder_fast --quick)
endif()
