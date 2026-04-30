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
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEQEC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-qec"

using namespace mlir;

/// \file
/// Strip QEC annotation ops (`qec.detector`, `qec.logical_observable`,
/// `qec.detectors_vectorized`) from the IR. These ops are inert from a
/// quantum-mechanical perspective: they declare relationships among prior
/// measurement results for analysis engines (DEM via Stim) but emit no gates
/// or measurements. Hardware targets and simulators that do not implement the
/// QEC runtime ABI must remove them before codegen so QIR emission does not
/// reference unimplemented `__quantum__qis__detector_*` runtime calls.
///
/// Mirrors `EraseNoise` in shape — pattern-based erasure invoked from
/// `createHardwareTargetPrepPipeline`. The emulation pipeline keeps these ops
/// so the analysis-engine path
/// (`cudaq_internal::analysis::computeDem`) can read them out of the recorded
/// Stim circuit.

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

class EraseQECPass : public cudaq::opt::impl::EraseQECBase<EraseQECPass> {
public:
  using EraseQECBase::EraseQECBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QEC erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseQECOpPattern<qec::DetectorOp>,
                    EraseQECOpPattern<qec::LogicalObservableOp>,
                    EraseQECOpPattern<qec::DetectorsVectorizedOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After QEC erasure:\n" << *op << "\n\n");
  }
};

} // namespace
