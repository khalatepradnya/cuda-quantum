/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERQECTOQIR
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lower-qec-to-qir"

using namespace mlir;

namespace {

/// Build a map from each `!quake.measure` SSA value to its chronological
/// measurement index. This walks all quake.mz (and mx, my) ops in program
/// order within each function and assigns sequential indices. Also returns
/// the total number of measurements found.
static std::pair<DenseMap<Value, std::int64_t>, std::int64_t>
buildMeasurementIndexMap(Operation *moduleOp) {
  DenseMap<Value, std::int64_t> measureMap;
  std::int64_t totalCount = 0;
  moduleOp->walk([&](func::FuncOp func) {
    std::int64_t counter = 0;
    func.walk([&](Operation *op) {
      if (isa<quake::MzOp, quake::MxOp, quake::MyOp>(op)) {
        for (auto result : op->getResults()) {
          if (isa<quake::MeasureType>(result.getType())) {
            measureMap[result] = counter++;
          }
        }
      }
    });
    totalCount = std::max(totalCount, counter);
  });
  return {measureMap, totalCount};
}

/// Lower a QEC op (detector or logical_observable) to a func.call that passes
/// measurement indices as a stack-allocated i64 array.
template <typename QECOp>
static void lowerQECOpToCall(QECOp op,
                             const DenseMap<Value, std::int64_t> &measureMap,
                             ModuleOp module, StringRef funcName,
                             SmallVector<Value> extraArgs = {}) {
  OpBuilder builder(op);
  auto loc = op.getLoc();
  auto i64Ty = builder.getI64Type();
  auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);

  auto measurements = op.getMeasurements();
  auto count = static_cast<std::int64_t>(measurements.size());

  if (count == 0) {
    op.erase();
    return;
  }

  auto countVal = builder.create<arith::ConstantIntOp>(loc, count, 64);
  Value arrayAlloc = builder.create<cudaq::cc::AllocaOp>(loc, i64Ty, countVal);

  for (auto [i, meas] : llvm::enumerate(measurements)) {
    auto it = measureMap.find(meas);
    std::int64_t measIdx =
        (it != measureMap.end()) ? it->second : static_cast<std::int64_t>(i);
    auto idxVal = builder.create<arith::ConstantIntOp>(loc, measIdx, 64);
    std::int32_t constIdx = static_cast<std::int32_t>(i);
    auto elemPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrI64Ty, arrayAlloc,
        ArrayRef<cudaq::cc::ComputePtrArg>{constIdx});
    builder.create<cudaq::cc::StoreOp>(loc, idxVal, elemPtr);
  }

  Value castPtr = builder.create<cudaq::cc::CastOp>(loc, ptrI64Ty, arrayAlloc);
  SmallVector<Value> callArgs = {castPtr, countVal};
  callArgs.append(extraArgs.begin(), extraArgs.end());

  SmallVector<Type> argTypes = {cudaq::cc::PointerType::get(i64Ty), i64Ty};
  for (auto v : extraArgs)
    argTypes.push_back(v.getType());

  auto funcTy =
      FunctionType::get(builder.getContext(), argTypes, /*results=*/{});
  auto funcDecl = cudaq::opt::factory::createFunction(
      funcName, funcTy.getResults(), funcTy.getInputs(), module);
  funcDecl.setPrivate();
  builder.create<func::CallOp>(loc, TypeRange{}, funcName, callArgs);
  op.erase();
}

class LowerQECToQIRPass
    : public cudaq::opt::impl::LowerQECToQIRBase<LowerQECToQIRPass> {
public:
  using LowerQECToQIRBase::LowerQECToQIRBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QEC lowering\n");

    auto [measureMap, totalMeasurements] = buildMeasurementIndexMap(moduleOp);

    // Collect ops first to avoid modifying while iterating
    SmallVector<qec::DetectorOp> detectorOps;
    SmallVector<qec::LogicalObservableOp> observableOps;
    SmallVector<qec::DetectorsVectorizedOp> vectorizedOps;
    moduleOp->walk([&](qec::DetectorOp op) { detectorOps.push_back(op); });
    moduleOp->walk(
        [&](qec::LogicalObservableOp op) { observableOps.push_back(op); });
    moduleOp->walk(
        [&](qec::DetectorsVectorizedOp op) { vectorizedOps.push_back(op); });

    auto module = dyn_cast<ModuleOp>(moduleOp);

    for (auto op : detectorOps) {
      OpBuilder builder(op);
      auto totalVal = builder.create<arith::ConstantIntOp>(
          op.getLoc(), totalMeasurements, 64);
      lowerQECOpToCall(op, measureMap, module, cudaq::opt::QIRDetectorIndices,
                       SmallVector<Value>{totalVal});
    }

    for (auto op : observableOps) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      auto totalVal =
          builder.create<arith::ConstantIntOp>(loc, totalMeasurements, 64);
      auto obsIdx = op.getObservableIndex();
      auto obsIdxVal = builder.create<arith::ConstantIntOp>(loc, obsIdx, 64);
      lowerQECOpToCall(op, measureMap, module,
                       cudaq::opt::QIRLogicalObservableIndices,
                       SmallVector<Value>{totalVal, obsIdxVal});
    }

    for (auto op : vectorizedOps)
      op.erase();

    LLVM_DEBUG(llvm::dbgs() << "After QEC lowering\n");
  }
};
} // namespace
