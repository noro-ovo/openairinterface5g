# SPDX-License-Identifier: LicenseRef-CSSL-1.0

message(STATUS "Add test nas_lib_test")

add_executable(nas_lib_test nas_lib_test.c)

add_dependencies(tests nas_lib_test)
add_test(NAME nas_lib_test COMMAND nas_lib_test)

target_link_libraries(nas_lib_test PRIVATE
    nr_nas
)

target_include_directories(nas_lib_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# UE NAS security mode control (includes nr_nas_msg.c for its static functions: not linked with nr_nas)
add_executable(nas_smc_test nas_smc_test.c)
add_dependencies(tests nas_smc_test)
add_test(NAME nas_smc_test COMMAND nas_smc_test)
target_link_libraries(nas_smc_test PRIVATE $<TARGET_PROPERTY:nr_nas,INTERFACE_LINK_LIBRARIES> ITTI lib_uicc nr_common)
target_include_directories(nas_smc_test PRIVATE $<TARGET_PROPERTY:nr_nas,INCLUDE_DIRECTORIES>)
