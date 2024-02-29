# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
)

set(PYCUDAQMLIR_TEST_PARAMS
  pycudaqmlir_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(PYCUDAQMLIR_TEST_DEPENDS
  _pycudaq _quakeDialect
  FileCheck)

add_custom_target(pycudaq-mlir-test-depends DEPENDS ${PYCUDAQ_TEST_DEPENDS})

add_lit_testsuite(check-py-cudaq-mlir "Running cudaq python regression tests."
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${PYCUDAQ_TEST_PARAMS}
  DEPENDS ${PYCUDAQ_TEST_DEPENDS}
)
set_target_properties(check-py-cudaq-mlir PROPERTIES FOLDER "Tools")

add_lit_testsuites(PYCUDAQMLIR ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${NVQPP_TEST_PARAMS}
  DEPENDS ${NVQPP_TEST_DEPENDS})

# Add nvqpp tests to the ctest suite
add_test(NAME py-cudaq-tests-mlir
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-py-cudaq-mlir
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})