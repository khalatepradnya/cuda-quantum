/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEQEC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-qec"

using namespace mlir;

namespace {

class EraseQECPass : public cudaq::opt::impl::EraseQECBase<EraseQECPass> {
public:
  using EraseQECBase::EraseQECBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "EraseQEC: before\n");
    SmallVector<Operation *> toErase;
    op->walk([&](Operation *inner) {
      if (isa<qec::DetectorOp, qec::LogicalObservableOp,
              qec::DetectorsVectorizedOp>(inner))
        toErase.push_back(inner);
    });
    for (auto *dead : toErase)
      dead->erase();
    LLVM_DEBUG(llvm::dbgs()
               << "EraseQEC: erased " << toErase.size() << " ops\n");
  }
};
} // namespace
