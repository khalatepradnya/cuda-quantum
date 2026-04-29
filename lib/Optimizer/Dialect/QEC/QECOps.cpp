/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"

using namespace mlir;

LogicalResult qec::DetectorsVectorizedOp::verify() {
  // Note: cc::StdvecType is dynamically sized at the type level, so any size
  // mismatch between prev and curr must be caught at runtime. The TableGen
  // operand constraint already enforces the element-type invariant; we
  // re-check it here as a defensive guard against future TableGen changes.
  auto prevTy = dyn_cast<cudaq::cc::StdvecType>(getPrev().getType());
  auto currTy = dyn_cast<cudaq::cc::StdvecType>(getCurr().getType());
  if (!prevTy || !currTy)
    return emitOpError(
        "prev and curr must both be !cc.stdvec<!cc.measure_handle>");
  if (!isa<cudaq::cc::MeasureHandleType>(prevTy.getElementType()) ||
      !isa<cudaq::cc::MeasureHandleType>(currTy.getElementType()))
    return emitOpError(
        "prev and curr stdvec element type must be !cc.measure_handle");
  return success();
}

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QEC/QECOps.cpp.inc"
