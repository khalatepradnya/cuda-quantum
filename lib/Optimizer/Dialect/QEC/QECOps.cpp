/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"

using namespace mlir;

// Note: `!cc.stdvec` carries no static size, so the previous size-matching
// check on `qec.detectors_vectorized` (which relied on the now-removed
// `!quake.measurements<N>`) cannot fire here. Backends and the `erase-qec`
// pass are responsible for the runtime size invariant; see followups.md for a
// future static-shape extension on `!cc.stdvec` if this becomes load-bearing.

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QEC/QECOps.cpp.inc"
