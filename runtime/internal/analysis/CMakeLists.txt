# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Note: the INTERFACE target `cudaq-analysis-headers` is defined in
# `runtime/CMakeLists.txt` so that `cudaq-common` / `nvqir` can pick up its
# include path before this subdirectory is configured. We just consume it here.

# Out-of-line implementation of the dem_policy CPO overload + result carrier.
# Lives in libcudaq-analysis so the simulator (which sees the policy via ADL)
# does not need to depend on stim headers.
add_library(cudaq-analysis
  SHARED
    Dem.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-analysis)

target_link_libraries(cudaq-analysis
  PUBLIC
    cudaq-analysis-headers
  PRIVATE
    cudaq-common
    cudaq-logger
    nvqir
    libstim
    fmt::fmt-header-only
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)

install(TARGETS cudaq-analysis
        EXPORT cudaq-analysis-targets
        DESTINATION lib)

install(EXPORT cudaq-analysis-targets
        FILE CUDAQAnalysisTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
