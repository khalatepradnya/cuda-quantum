/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QIRPATTERNANALYSIS
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "qir-pattern-analysis"

using namespace mlir;

namespace {
struct AllocaMeasureStoreAnalysis {
  AllocaMeasureStoreAnalysis() = default;

  // Walk a function to identify all the measure-discriminate-store patterns and
  // collect the associated AllocaOps.
  // Collect only unique AllocaOps - since each may correspond to multiple
  // measurement operations

  explicit AllocaMeasureStoreAnalysis(func::FuncOp func) {
    DenseMap<Value, Operation *> valueToMeasurement;
    llvm::SmallSet<cudaq::cc::AllocaOp, 4> uniqueAllocaOps;

    // First pass: identify measurements and propagate through uses
    func.walk([&](Operation *op) {
      if (op->hasTrait<cudaq::QuantumMeasure>()) {
        for (auto result : op->getResults())
          valueToMeasurement[result] = op;
      } else if (auto discrOp = dyn_cast<quake::DiscriminateOp>(op)) {
        if (valueToMeasurement.count(discrOp.getMeasurement()))
          valueToMeasurement[discrOp.getResult()] =
              valueToMeasurement[discrOp.getMeasurement()];
      } else if (auto castOp = dyn_cast<cudaq::cc::CastOp>(op)) {
        if (valueToMeasurement.count(castOp.getValue()))
          valueToMeasurement[castOp.getResult()] =
              valueToMeasurement[castOp.getValue()];
      }
    });

    // Second pass: find stores of measurement values and trace to allocas
    func.walk([&](cudaq::cc::StoreOp storeOp) {
      if (valueToMeasurement.count(storeOp.getValue())) {
        Value ptr = storeOp.getPtrvalue();
        while (ptr) {
          if (auto allocaOp =
                  dyn_cast<cudaq::cc::AllocaOp>(ptr.getDefiningOp())) {
            uniqueAllocaOps.insert(allocaOp);
            break;
          }
          if (auto castOp = dyn_cast<cudaq::cc::CastOp>(ptr.getDefiningOp())) {
            ptr = castOp.getValue();
            continue;
          }
          if (auto computePtrOp =
                  dyn_cast<cudaq::cc::ComputePtrOp>(ptr.getDefiningOp())) {
            ptr = computePtrOp.getBase();
            continue;
          }
          break;
        }
      }
    });

    allocaOps.append(uniqueAllocaOps.begin(), uniqueAllocaOps.end());
  }

  SmallVector<cudaq::cc::AllocaOp> allocaOps;
};

LogicalResult
insertArrayRecordingCalls(func::FuncOp funcOp,
                          const SmallVector<cudaq::cc::AllocaOp> &allocaOps) {
  size_t resultCount = 0;
  for (auto alloca : allocaOps) {
    if (auto arrType =
            alloca.getElementType().dyn_cast<cudaq::cc::ArrayType>()) {
      resultCount += arrType.getSize();
    } else {
      resultCount += 1;
    }
  }
  if (resultCount == 0)
    return success();

  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  mlir::Location loc = funcOp.getLoc();
  builder.setInsertionPointAfter(allocaOps[0]);

  std::string labelStr = "array<i8 x " + std::to_string(resultCount) + ">";
  auto strLitTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(
      builder.getContext(), builder.getI8Type(), labelStr.size() + 1));
  Value lit = builder.create<cudaq::cc::CreateStringLiteralOp>(
      loc, strLitTy, builder.getStringAttr(labelStr));
  auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
  Value label = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, lit);
  Value size = builder.create<arith::ConstantIntOp>(loc, resultCount, 64);
  builder.create<func::CallOp>(loc, TypeRange{},
                               cudaq::opt::QIRArrayRecordOutput,
                               ArrayRef<Value>{size, label});

  // Also add the declaration to the module
  auto module = funcOp->getParentOfType<ModuleOp>();
  auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
  if (failed(irBuilder.loadIntrinsic(funcOp->getParentOfType<ModuleOp>(),
                                     cudaq::opt::QIRArrayRecordOutput))) {
    return failure();
  }

  return success();
}

struct QIRPatternAnalysisPass
    : public cudaq::opt::impl::QIRPatternAnalysisBase<QIRPatternAnalysisPass> {

  using QIRPatternAnalysisBase::QIRPatternAnalysisBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;
    if (!funcOp->hasAttr(cudaq::entryPointAttrName))
      return;
    if (funcOp->hasAttr(cudaq::runtime::enableCudaqRun))
      return;

    AllocaMeasureStoreAnalysis analysis(funcOp);
    if (analysis.allocaOps.empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "Before adding array recording call:\n"
                            << *funcOp);
    if (failed(insertArrayRecordingCalls(funcOp, analysis.allocaOps)))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After adding array recording call:\n"
                            << *funcOp);
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createQIRPatternAnalysis() {
  return std::make_unique<QIRPatternAnalysisPass>();
}
