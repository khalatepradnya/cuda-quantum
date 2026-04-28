/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// Only an individual qubit measurement returns a scalar token. Both
// `!quake.measure` and `!cc.measure_handle` are scalar per-qubit measurement
// results, so neither requires expansion to a register.
template <typename A>
bool usesIndividualQubit(A x) {
  return isa<quake::MeasureType, cudaq::cc::MeasureHandleType>(x.getType());
}

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
//
// Handles both result-type families that the vector form of `quake.mz`/`mx`/
// `my` can carry:
//   - `!cc.stdvec<!quake.measure>` -- the legacy form. The only legitimate
//     consumer is `quake.discriminate`, so the rewrite folds the per-element
//     measurements straight into a `cc.stdvec_init -> !cc.stdvec<i1>`.
//   - `!cc.stdvec<!cc.measure_handle>` -- the handle-vector value can have
//   non-discriminate consumers  Those consumers
//     expect a value of the original handle-stdvec type, so the rewrite
//     additionally builds a per-element handle buffer and folds it into a
//     `cc.stdvec_init -> !cc.stdvec<!cc.measure_handle>` that replaces all
//     remaining uses.
template <typename A>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    auto loc = measureOp.getLoc();
    auto *ctx = rewriter.getContext();

    // The dynamic-legality predicate filters out the scalar forms, so by
    // construction the result type here is `!cc.stdvec<X>` for some X.
    auto stdvecResTy =
        dyn_cast<cudaq::cc::StdvecType>(measureOp.getMeasOut().getType());
    auto handleTy = cudaq::cc::MeasureHandleType::get(ctx);
    bool isHandleResult =
        isa<cudaq::cc::MeasureHandleType>(stdvecResTy.getElementType());

    // Per-element scalar result type tracks the original stdvec element
    // type. For handle inputs we measure into `!cc.measure_handle` per
    // qubit.
    Type perElemTy = isHandleResult
                         ? static_cast<Type>(handleTy)
                         : static_cast<Type>(quake::MeasureType::get(ctx));

    // Classify users so we only allocate the buffers we actually need.
    // The legacy `!quake.measure` path has only `quake.discriminate`
    // consumers by construction; the handle path may have either, both,
    // or none.
    bool hasDiscUser = false;
    bool hasNonDiscUser = false;
    for (auto *u : measureOp.getMeasOut().getUsers()) {
      if (isa<quake::DiscriminateOp>(u))
        hasDiscUser = true;
      else
        hasNonDiscUser = true;
    }
    // Allocation policy:
    //   - Legacy `!cc.stdvec<!quake.measure>` always allocates the i1 buffer.
    //   - `!cc.stdvec<!cc.measure_handle>` allocates each buffer only when a
    //   consumer in that element-type class is present.
    bool needI1Buf = !isHandleResult || hasDiscUser;
    bool needHandleBuf = isHandleResult && hasNonDiscUser;

    // 1. Determine the total number of qubits we need to measure. This
    // determines the size of the buffer of bools to create to store the results
    // in.
    unsigned numQubits = 0u;
    for (auto v : measureOp.getTargets())
      if (v.getType().template isa<quake::RefType>())
        ++numQubits;
    Value totalToRead =
        rewriter.template create<arith::ConstantIntOp>(loc, numQubits, 64);
    auto i64Ty = rewriter.getI64Type();
    for (auto v : measureOp.getTargets())
      if (v.getType().template isa<quake::VeqType>()) {
        Value vecSz = rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, v);
        totalToRead =
            rewriter.template create<arith::AddIOp>(loc, totalToRead, vecSz);
      }

    // 2. Create the buffers (one per output kind we actually need).
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value i1Buff;
    if (needI1Buf)
      i1Buff =
          rewriter.template create<cudaq::cc::AllocaOp>(loc, i8Ty, totalToRead);
    Value handleBuff;
    if (needHandleBuf)
      handleBuff = rewriter.template create<cudaq::cc::AllocaOp>(loc, handleTy,
                                                                 totalToRead);

    // Per-element store helper. Each qubit is measured exactly once with
    // `perElemTy`; the resulting value is fanned out to whichever buffers we
    // allocated (i1 for discriminate consumers, handle for non-discriminate
    // consumers).
    auto storePerElement = [&](OpBuilder &builder, Location loc, Value meas,
                               Value offset) {
      if (needI1Buf) {
        auto bit =
            builder.template create<quake::DiscriminateOp>(loc, i1Ty, meas);
        auto addr = builder.template create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i8Ty), i1Buff, offset);
        auto bitByte = builder.template create<cudaq::cc::CastOp>(
            loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        builder.template create<cudaq::cc::StoreOp>(loc, bitByte, addr);
      }
      if (needHandleBuf) {
        auto addr = builder.template create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(handleTy), handleBuff, offset);
        builder.template create<cudaq::cc::StoreOp>(loc, meas, addr);
      }
    };

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers, loop over the entire set of qubits.
    Value buffOff = rewriter.template create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.template create<arith::ConstantIntOp>(loc, 1, 64);
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        auto meas = rewriter.template create<A>(loc, perElemTy, v).getMeasOut();
        storePerElement(rewriter, loc, meas, buffOff);
        buffOff = rewriter.template create<arith::AddIOp>(loc, buffOff, one);
      } else {
        assert(isa<quake::VeqType>(v.getType()));
        Value vecSz = rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, v);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, vecSz,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value iv = block.getArgument(0);
              Value qv =
                  builder.template create<quake::ExtractRefOp>(loc, v, iv);
              auto meas = builder.template create<A>(loc, perElemTy, qv);
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              Value offset =
                  builder.template create<arith::AddIOp>(loc, iv, buffOff);
              storePerElement(builder, loc, meas.getMeasOut(), offset);
            });
        buffOff = rewriter.template create<arith::AddIOp>(loc, buffOff, vecSz);
      }
    }

    // 4. Replace each `quake.discriminate` consumer with a
    // `cc.stdvec_init -> !cc.stdvec<i1>` over the i1 buffer.
    if (needI1Buf) {
      auto stdvecI1Ty = cudaq::cc::StdvecType::get(ctx, i1Ty);
      auto ptrArrI1Ty =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
      for (auto *out : llvm::to_vector(measureOp.getMeasOut().getUsers()))
        if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
          auto buffCast = rewriter.template create<cudaq::cc::CastOp>(
              loc, ptrArrI1Ty, i1Buff);
          rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
              disc, stdvecI1Ty, buffCast, totalToRead);
        }
    }

    // 5. For the handle path with non-discriminate consumers, build a
    // `cc.stdvec_init -> !cc.stdvec<!cc.measure_handle>` over the handle
    // buffer and route the original result's remaining users to it via
    // `replaceOp` (one atomic substitution)
    Value replacementVal;
    if (needHandleBuf) {
      auto stdvecHandleTy = cudaq::cc::StdvecType::get(ctx, handleTy);
      auto handleStdvec = rewriter.template create<cudaq::cc::StdvecInitOp>(
          loc, stdvecHandleTy, handleBuff, totalToRead);
      replacementVal = handleStdvec.getResult();
    }

    SmallVector<Value> replacements;
    replacements.push_back(replacementVal);
    for (auto wire : measureOp.getWires()) {
      (void)wire;
      replacements.push_back(nullptr);
    }
    rewriter.replaceOp(measureOp, replacements);
    return success();
  }
};

namespace {
using MxRewrite = ExpandRewritePattern<quake::MxOp>;
using MyRewrite = ExpandRewritePattern<quake::MyOp>;
using MzRewrite = ExpandRewritePattern<quake::MzOp>;

/// Convert a `quake.reset` with a `veq` argument into a loop over the elements
/// of the `veq` and `quake.reset` on each of them.
class ResetRewrite : public OpRewritePattern<quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ResetOp resetOp,
                                PatternRewriter &rewriter) const override {
    auto loc = resetOp.getLoc();
    auto veqArg = resetOp.getTargets();
    auto i64Ty = rewriter.getI64Type();
    Value vecSz = rewriter.create<quake::VeqSizeOp>(loc, i64Ty, veqArg);
    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSz,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value qv = builder.create<quake::ExtractRefOp>(loc, veqArg, iv);
          builder.create<quake::ResetOp>(loc, TypeRange{}, qv);
        });
    rewriter.eraseOp(resetOp);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<MxRewrite, MyRewrite, MzRewrite, ResetRewrite>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<quake::MxOp>(
        [](quake::MxOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MyOp>(
        [](quake::MyOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MzOp>(
        [](quake::MzOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::ResetOp>([](quake::ResetOp r) {
      return !isa<quake::VeqType>(r.getTargets().getType());
    });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitOpError("could not expand measurements");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createExpandMeasurementsPass() {
  return std::make_unique<ExpandMeasurementsPass>();
}
