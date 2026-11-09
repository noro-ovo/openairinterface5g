add_executable(test_lut_dmrs test_lut_dmrs.c)
target_link_libraries(test_lut_dmrs PRIVATE
    MAC_NR_COMMON
    nr_common
    asn1_nr_rrc_hdrs
    minimal_lib m)
add_dependencies(tests test_lut_dmrs)
add_test(NAME test_lut_dmrs COMMAND test_lut_dmrs)

add_executable(test_pdcch_candidates test_pdcch_candidates.c)
target_link_libraries(test_pdcch_candidates PRIVATE
    MAC_NR_COMMON
    nr_common
    asn1_nr_rrc_hdrs
    minimal_lib m)
add_dependencies(tests test_pdcch_candidates)
add_test(NAME test_pdcch_candidates COMMAND test_pdcch_candidates)
