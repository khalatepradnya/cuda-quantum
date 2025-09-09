# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-deprecated-declarations")

pybind11_add_module(cudaq_qed SHARED py_error_detect.cpp)
target_include_directories(cudaq_qed PRIVATE 
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(cudaq_qed PRIVATE cudaq cudaq-qed-simulator)
set_target_properties(cudaq_qed PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/cudaq/qed)
install(TARGETS cudaq_qed DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/cudaq/qed)
