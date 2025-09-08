# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Create two libraries, one for the simulator and one for the QPU

# Create the simulator library
add_library(cudaq-qed-simulator SHARED)
set_target_properties(cudaq-qed-simulator PROPERTIES LINKER_LANGUAGE CXX)

set(SIMULATOR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/simulator.cpp")
set(SIMULATOR_OBJ "${CMAKE_CURRENT_BINARY_DIR}/simulator.o")

# Create a custom command to compile simulator.cpp with nvq++
# Ref: https://github.com/NVIDIA/cudaqx/blob/719427c879cd28eba50263856ae0e39b0a656f9f/cmake/Modules/CUDA-QX.cmake

add_custom_command(
    OUTPUT ${SIMULATOR_OBJ}
    COMMAND ${CMAKE_BINARY_DIR}/bin/nvq++
        -std=c++20 -c -fPIC --enable-mlir
        -I${CMAKE_SOURCE_DIR}/include
        -I${CMAKE_SOURCE_DIR}/runtime
        -I${CMAKE_BINARY_DIR}/include
        ${SIMULATOR_SRC} -o simulator.o
    DEPENDS ${SIMULATOR_SRC}
    COMMENT "Compiling simulator.cpp with nvq++"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

target_sources(cudaq-qed-simulator PRIVATE ${SIMULATOR_OBJ})
add_dependencies(cudaq-qed-simulator cudaq nvq++)
target_link_libraries(cudaq-qed-simulator PRIVATE cudaq)

install(TARGETS cudaq-qed-simulator DESTINATION lib)
