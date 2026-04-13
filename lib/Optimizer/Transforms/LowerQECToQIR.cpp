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
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
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

/// Count how many individual measurements a single mz/mx/my op produces.
/// A single-qubit measurement returns one !quake.measure.
/// A multi-qubit measurement returns !cc.stdvec<!quake.measure>; the count
/// equals the number of qubits measured (derived from the veq operand size
/// when statically known, otherwise from the number of qubit operands).
static std::int64_t countMeasurements(Operation *op) {
  std::int64_t count = 0;
  for (auto result : op->getResults()) {
    if (isa<quake::MeasureType>(result.getType())) {
      count++;
    } else if (auto stdvecTy =
                   dyn_cast<cudaq::cc::StdvecType>(result.getType())) {
      if (isa<quake::MeasureType>(stdvecTy.getElementType())) {
        // Try to determine the number of qubits from the veq operand.
        for (auto operand : op->getOperands()) {
          if (auto veqTy = dyn_cast<quake::VeqType>(operand.getType())) {
            if (veqTy.hasSpecifiedSize()) {
              count += veqTy.getSize();
              break;
            }
          }
        }
        if (count == 0)
          count = 1; // Conservative: at least one measurement
      }
    }
  }
  return count;
}

/// Try to resolve a `!quake.measure` SSA value that was extracted from a
/// multi-qubit measurement vector back to its chronological index.
/// The IR pattern is: quake.mz -> cc.stdvec_data -> cc.compute_ptr[idx]
/// -> cc.load -> !quake.measure. Returns std::nullopt if the chain cannot
/// be traced.
static std::optional<std::int64_t>
traceVectorElement(Value measVal,
                   const DenseMap<Value, std::int64_t> &vecBaseMap) {
  auto *defOp = measVal.getDefiningOp();
  if (!defOp || !isa<cudaq::cc::LoadOp>(defOp))
    return std::nullopt;
  auto loadOp = cast<cudaq::cc::LoadOp>(defOp);

  auto *ptrOp = loadOp.getPtrvalue().getDefiningOp();
  if (!ptrOp || !isa<cudaq::cc::ComputePtrOp>(ptrOp))
    return std::nullopt;
  auto computePtr = cast<cudaq::cc::ComputePtrOp>(ptrOp);

  // Extract the constant index from the compute_ptr rawConstantIndices.
  auto indices = computePtr.getRawConstantIndices();
  if (indices.empty())
    return std::nullopt;
  std::int64_t elemIdx = indices[0];
  if (elemIdx < 0)
    return std::nullopt; // Dynamic index, cannot resolve at compile time

  auto *dataOp = computePtr.getBase().getDefiningOp();
  if (!dataOp || !isa<cudaq::cc::StdvecDataOp>(dataOp))
    return std::nullopt;
  auto stdvecData = cast<cudaq::cc::StdvecDataOp>(dataOp);

  Value vecVal = stdvecData.getOperand();
  auto it = vecBaseMap.find(vecVal);
  if (it == vecBaseMap.end())
    return std::nullopt;

  return it->second + elemIdx;
}

/// Build a map from each `!quake.measure` SSA value to its chronological
/// measurement index. This walks all quake.mz (and mx, my) ops in program
/// order within each function and assigns sequential indices.
///
/// For multi-qubit measurements that return `!cc.stdvec<!quake.measure>`,
/// the base index is recorded so that individual elements (extracted via
/// cc.stdvec_data + cc.compute_ptr + cc.load) can be resolved by
/// traceVectorElement().
static std::pair<DenseMap<Value, std::int64_t>, std::int64_t>
buildMeasurementIndexMap(Operation *moduleOp) {
  DenseMap<Value, std::int64_t> measureMap;
  DenseMap<Value, std::int64_t> vecBaseMap;
  std::int64_t totalCount = 0;
  moduleOp->walk([&](func::FuncOp func) {
    std::int64_t counter = 0;
    func.walk([&](Operation *op) {
      if (isa<quake::MzOp, quake::MxOp, quake::MyOp>(op)) {
        for (auto result : op->getResults()) {
          if (isa<quake::MeasureType>(result.getType())) {
            measureMap[result] = counter++;
          } else if (auto stdvecTy =
                         dyn_cast<cudaq::cc::StdvecType>(result.getType())) {
            if (isa<quake::MeasureType>(stdvecTy.getElementType())) {
              vecBaseMap[result] = counter;
              counter += countMeasurements(op);
            }
          }
        }
      }
    });
    totalCount = std::max(totalCount, counter);
  });

  // Resolve cc.load values that trace back to multi-qubit measurement vectors.
  moduleOp->walk([&](cudaq::cc::LoadOp loadOp) {
    Value result = loadOp.getResult();
    if (!isa<quake::MeasureType>(result.getType()))
      return;
    if (measureMap.count(result))
      return; // Already directly mapped
    if (auto idx = traceVectorElement(result, vecBaseMap))
      measureMap[result] = *idx;
  });

  return {measureMap, totalCount};
}

/// Lower a QEC op (detector or logical_observable) to a func.call that passes
/// measurement indices as a stack-allocated i64 array. Returns failure if any
/// measurement operand cannot be resolved to a chronological index.
template <typename QECOp>
static LogicalResult
lowerQECOpToCall(QECOp op, const DenseMap<Value, std::int64_t> &measureMap,
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
    return success();
  }

  auto countVal = builder.create<arith::ConstantIntOp>(loc, count, 64);
  Value arrayAlloc = builder.create<cudaq::cc::AllocaOp>(loc, i64Ty, countVal);

  for (auto [i, meas] : llvm::enumerate(measurements)) {
    auto it = measureMap.find(meas);
    if (it == measureMap.end()) {
      op.emitOpError("measurement operand #")
          << i << " could not be resolved to a chronological measurement "
          << "index. Ensure it originates from a quake.mz/mx/my operation.";
      return failure();
    }
    auto idxVal = builder.create<arith::ConstantIntOp>(loc, it->second, 64);
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
  return success();
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
    bool failed = false;

    for (auto op : detectorOps) {
      OpBuilder builder(op);
      auto totalVal = builder.create<arith::ConstantIntOp>(
          op.getLoc(), totalMeasurements, 64);
      if (lowerQECOpToCall(op, measureMap, module,
                           cudaq::opt::QIRDetectorIndices,
                           SmallVector<Value>{totalVal})
              .failed())
        failed = true;
    }

    for (auto op : observableOps) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      auto totalVal =
          builder.create<arith::ConstantIntOp>(loc, totalMeasurements, 64);
      auto obsIdx = op.getObservableIndex();
      auto obsIdxVal = builder.create<arith::ConstantIntOp>(loc, obsIdx, 64);
      if (lowerQECOpToCall(op, measureMap, module,
                           cudaq::opt::QIRLogicalObservableIndices,
                           SmallVector<Value>{totalVal, obsIdxVal})
              .failed())
        failed = true;
    }

    // detectors_vectorized takes two cc.stdvec<!quake.measure> operands whose
    // element count is only known at runtime. Full lowering requires emitting
    // a loop over the vector elements at the IR level, or a dedicated runtime
    // function that accepts two stdvec pointers. For the prototype, emit an
    // error if this op appears in compiled mode (library mode dispatches
    // directly and never reaches this pass).
    for (auto op : vectorizedOps) {
      op.emitWarning("qec.detectors_vectorized is not yet lowered in compiled "
                     "mode; the op will be erased. Use individual "
                     "cudaq::detector() calls for compiled-mode support.");
      op.erase();
    }

    if (failed)
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "After QEC lowering\n");
  }
};
} // namespace
