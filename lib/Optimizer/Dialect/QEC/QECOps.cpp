/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"

using namespace mlir;

LogicalResult qec::DetectorsVectorizedOp::verify() {
  auto prevTy = cast<quake::MeasurementsType>(getPrev().getType());
  auto currTy = cast<quake::MeasurementsType>(getCurr().getType());
  if (prevTy.hasSpecifiedSize() && currTy.hasSpecifiedSize() &&
      prevTy.getSize() != currTy.getSize())
    return emitOpError("prev and curr measurement vectors must have matching "
                       "sizes, got ")
           << prevTy.getSize() << " and " << currTy.getSize();
  return success();
}

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QEC/QECOps.cpp.inc"
