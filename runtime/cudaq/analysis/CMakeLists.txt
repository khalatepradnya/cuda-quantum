# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# DEM (Detector Error Model) analysis primitive. Implemented in its own
# shared library because `Dem.cpp` includes Stim's headers and links
# `libstim`; folding it into the main `cudaq` library would force every
# CUDA-Q consumer to pull Stim transitively. The user-facing header
# `dem.h` is the public surface (`cudaq::analysis::compute_dem`) and
# rides through the `runtime/cudaq` install rule. Linking `nvqir`
# privately picks up `nvqir::AnalysisScope`, the RAII type that backs
# `dem::make_scope`.
add_library(cudaq-analysis SHARED Dem.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-analysis)

target_include_directories(cudaq-analysis
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
)

target_link_libraries(cudaq-analysis
  PUBLIC
    cudaq-common
  PRIVATE
    cudaq-logger
    cudaq-mlir-runtime-headers
    nvqir
    libstim
    fmt::fmt-header-only
)

install(TARGETS cudaq-analysis EXPORT cudaq-targets DESTINATION lib)
