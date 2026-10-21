/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Declares `phys-to-quake`, which lowers a P3 physical graph to Quake. It
// needs CUDA-Q's Quake and CC dialects, so it is built only when
// `QLX_USE_CUDAQ_SDK` is ON, the default (`QLX_HAS_CUDAQ_QUAKE`).

#ifndef QLX_CONVERSION_PHYSTOQUAKE_PASSES_H
#define QLX_CONVERSION_PHYSTOQUAKE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace qlx {

#define GEN_PASS_DECL
#include "qlx/Conversion/PhysToQuakePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Conversion/PhysToQuakePasses.h.inc"

} // namespace qlx

#endif // QLX_CONVERSION_PHYSTOQUAKE_PASSES_H
