# SPDX-License-Identifier: LicenseRef-CSSL-1.0

# nr_macbench: gNB MAC scheduler without PHY, FAPI loopback, conformance checker on every slot.
add_executable(nr_macbench macbench.c)
target_link_libraries(nr_macbench PRIVATE
  -Wl,--start-group UTIL SIMU PHY_NR_COMMON PHY_NR SCHED_NR_LIB MAC_NR_COMMON CONFIG_LIB L2_NR -Wl,--end-group
  m pthread ${T_LIB} ITTI dl physim_common time_management_core
  softmodem_common
  )
target_link_libraries(nr_macbench PRIVATE asn1_nr_rrc_hdrs asn1_lte_rrc_hdrs)

if(ENABLE_TESTS)
  add_dependencies(tests nr_macbench)
endif()
