# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Headers-only INTERFACE target for the analysis-engine module. Registered
# here (rather than inside `internal/analysis/CMakeLists.txt`) because
# `cudaq/algorithms/policy_dispatch.h` includes
# `cudaq_internal/analysis/DemPolicy.h` and is consumed by `cudaq-common` /
# `nvqir`, which are configured before `internal/`. The matching `.cpp`
# library `cudaq-analysis` is registered later via `internal/analysis/`,
# which can come after `nvqir` because libstim is registered through
# `nvqir/stim`.
add_library(cudaq-analysis-headers INTERFACE)
target_include_directories(cudaq-analysis-headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/analysis/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS cudaq-analysis-headers EXPORT cudaq-analysis-targets)

add_subdirectory(logger)
add_subdirectory(common)
add_subdirectory(nvqir)
# `internal` registers the compiler runtime (cudaq-mlir-runtime) and the
# analysis-engine module (cudaq-analysis). It must come after `nvqir` because
# the analysis subdir consumes `libstim`, which is registered transitively
# through `nvqir/stim`.
add_subdirectory(internal)
add_subdirectory(cudaq)
add_subdirectory(test)

# Install
# ==============================================================================

install (DIRECTORY cudaq DESTINATION include 
            FILES_MATCHING PATTERN "*.h" 
            PATTERN "nlopt-src" EXCLUDE)
install (DIRECTORY include/cudaq DESTINATION include FILES_MATCHING PATTERN "*.h")
install (DIRECTORY common DESTINATION include FILES_MATCHING PATTERN "*.h")
install (FILES nvqir/CircuitSimulator.h
               nvqir/QIRTypes.h
               nvqir/Gates.h
        DESTINATION include/nvqir)
install (FILES cudaq.h DESTINATION include)
