/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEQECOPS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-qec-ops"

using namespace mlir;

namespace {
template <typename Op>
class EraseQECOpPattern : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseQECOpsPass
    : public cudaq::opt::impl::EraseQECOpsBase<EraseQECOpsPass> {
public:
  using EraseQECOpsBase::EraseQECOpsBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QEC erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseQECOpPattern<qec::DetectorOp>,
                    EraseQECOpPattern<qec::DetectorsVectorizedOp>,
                    EraseQECOpPattern<qec::LogicalObservableOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After QEC erasure:\n" << *op << "\n\n");
  }
};
} // namespace
