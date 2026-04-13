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
/// A single-qubit measurement returns one `!quake.measure`.
/// A multi-qubit measurement returns `!quake.measurements<N>`; the count
/// equals N when statically known, or can be derived from the veq operand.
///
/// Returns 0 and emits a diagnostic if the measurement count cannot be
/// determined statically. The previous fallback of returning 1 for
/// dynamically-sized veq<?> silently produced wrong measurement indices
/// for all subsequent measurements, breaking detector-to-measurement
/// associations.
static std::int64_t countMeasurements(Operation *op) {
  std::int64_t count = 0;
  for (auto result : op->getResults()) {
    if (isa<quake::MeasureType>(result.getType())) {
      count++;
    } else if (auto measTy =
                   dyn_cast<quake::MeasurementsType>(result.getType())) {
      // !quake.measurements<N> has a known size directly on the type.
      if (measTy.hasSpecifiedSize()) {
        count += measTy.getSize();
      } else {
        // !quake.measurements<?> -- try the veq operand as a fallback.
        bool resolved = false;
        for (auto operand : op->getOperands()) {
          if (auto veqTy = dyn_cast<quake::VeqType>(operand.getType())) {
            if (veqTy.hasSpecifiedSize()) {
              count += veqTy.getSize();
              resolved = true;
              break;
            }
          }
        }
        // Dynamically-sized veq<?> cannot be resolved at compile time.
        // Returning a wrong count (e.g. 1) would silently shift all
        // subsequent measurement indices, producing incorrect rec[-N]
        // lookbacks in the Stim backend. Emit a diagnostic instead.
        if (!resolved) {
          op->emitError("measurement on dynamically-sized qubit register "
                        "(veq<?>) cannot be resolved to a static count. "
                        "QEC detector lowering requires statically-known "
                        "measurement counts.");
          return 0;
        }
      }
    }
  }
  return count;
}

/// Try to resolve a `!quake.measure` SSA value that was extracted from a
/// multi-qubit measurement collection back to its chronological index.
///
/// Two IR patterns are supported:
///   1. quake.get_measure %ms[i] : the native Quake extraction op
///   2. cc.stdvec_data -> cc.compute_ptr[idx] -> cc.load : legacy CC pattern
///      that appears if an earlier pass lowered !quake.measurements to cc types
///
/// Returns std::nullopt if the chain cannot be traced.
static std::optional<std::int64_t>
traceVectorElement(Value measVal,
                   const DenseMap<Value, std::int64_t> &vecBaseMap) {
  auto *defOp = measVal.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  // Pattern 1: quake.get_measure %ms[i]
  if (auto getMeas = dyn_cast<quake::GetMeasureOp>(defOp)) {
    auto rawIdx = getMeas.getRawIndex();
    if (rawIdx == quake::GetMeasureOp::kDynamicIndex)
      return std::nullopt; // Dynamic index, cannot resolve at compile time
    Value msVal = getMeas.getMeasurements();
    auto it = vecBaseMap.find(msVal);
    if (it == vecBaseMap.end())
      return std::nullopt;
    return it->second + static_cast<std::int64_t>(rawIdx);
  }

  // Pattern 2: cc.load <- cc.compute_ptr <- cc.stdvec_data (legacy CC path)
  if (isa<cudaq::cc::LoadOp>(defOp)) {
    auto loadOp = cast<cudaq::cc::LoadOp>(defOp);
    auto *ptrOp = loadOp.getPtrvalue().getDefiningOp();
    if (!ptrOp || !isa<cudaq::cc::ComputePtrOp>(ptrOp))
      return std::nullopt;
    auto computePtr = cast<cudaq::cc::ComputePtrOp>(ptrOp);
    auto indices = computePtr.getRawConstantIndices();
    if (indices.empty())
      return std::nullopt;
    std::int64_t elemIdx = indices[0];
    if (elemIdx < 0)
      return std::nullopt;
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

  return std::nullopt;
}

/// Per-function measurement analysis: the measurement index map and the total
/// count of measurements in that function. Each function's measurements are
/// numbered independently starting from 0, matching QIR semantics where each
/// function executes in its own context.
struct FuncMeasurementInfo {
  DenseMap<Value, std::int64_t> measureMap;
  // Base index of each multi-qubit measurement collection (!quake.measurements
  // result of quake.mz on veq). Used to expand !quake.measurements<N> operands
  // to N contiguous indices, and to resolve quake.get_measure extractions.
  DenseMap<Value, std::int64_t> vecBaseMap;
  std::int64_t totalMeasurements = 0;
  bool hadError = false;
};

/// Build a measurement index map for a single function. Walks all quake.mz
/// (and mx, my) ops in program order and assigns sequential indices.
///
/// The previous implementation computed a single module-wide totalMeasurements
/// as max(per-function counts). This was wrong: if function A has 5
/// measurements and function B has 3, function B's detectors would receive
/// totalMeasurements=5, producing incorrect rec[-N] lookbacks in the Stim
/// backend. Each function must use its own measurement count.
static FuncMeasurementInfo buildMeasurementIndexMapForFunc(func::FuncOp func) {
  FuncMeasurementInfo info;
  std::int64_t counter = 0;

  func.walk([&](Operation *op) {
    if (isa<quake::MzOp, quake::MxOp, quake::MyOp>(op)) {
      for (auto result : op->getResults()) {
        if (isa<quake::MeasureType>(result.getType())) {
          info.measureMap[result] = counter++;
        } else if (isa<quake::MeasurementsType>(result.getType())) {
          // Multi-qubit mz produces !quake.measurements<N>. Record the
          // base index so that quake.get_measure extractions can be
          // resolved to absolute indices by traceVectorElement().
          info.vecBaseMap[result] = counter;
          auto measCount = countMeasurements(op);
          if (measCount == 0)
            info.hadError = true;
          counter += measCount;
        }
      }
    }
  });

  // Resolve individual !quake.measure values extracted from multi-qubit
  // measurement collections. Handles both:
  //   - quake.get_measure %ms[i] (native Quake pattern)
  //   - cc.load <- cc.compute_ptr <- cc.stdvec_data (legacy CC pattern)
  func.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!isa<quake::MeasureType>(result.getType()))
        continue;
      if (info.measureMap.count(result))
        continue;
      if (auto idx = traceVectorElement(result, info.vecBaseMap))
        info.measureMap[result] = *idx;
    }
  });

  info.totalMeasurements = counter;
  return info;
}

/// Lower a QEC op (detector or logical_observable) to a func.call that passes
/// measurement indices as a stack-allocated i64 array. Returns failure if any
/// measurement operand cannot be resolved to a chronological index.
///
/// Operands may be individual `!quake.measure` values (looked up directly in
/// measureMap) or `!quake.measurements<N>` collections (expanded into N
/// consecutive indices starting from the base in vecBaseMap). This handles
/// both the variadic template API and the vector overload.
template <typename QECOp>
static LogicalResult
lowerQECOpToCall(QECOp op, const FuncMeasurementInfo &info,
                 ModuleOp module, StringRef funcName,
                 SmallVector<Value> extraArgs = {}) {
  OpBuilder builder(op);
  auto loc = op.getLoc();
  auto i64Ty = builder.getI64Type();
  auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);

  // Flatten operands: each !quake.measure contributes 1 index, each
  // !quake.measurements<N> contributes N contiguous indices.
  SmallVector<std::int64_t> indices;
  for (auto [i, meas] : llvm::enumerate(op.getMeasurements())) {
    if (isa<quake::MeasureType>(meas.getType())) {
      auto it = info.measureMap.find(meas);
      if (it == info.measureMap.end()) {
        op.emitOpError("measurement operand #")
            << i << " could not be resolved to a chronological measurement "
            << "index. Ensure it originates from a quake.mz/mx/my operation.";
        return failure();
      }
      indices.push_back(it->second);
    } else if (auto measTy =
                   dyn_cast<quake::MeasurementsType>(meas.getType())) {
      // A !quake.measurements<N> operand expands to N consecutive indices.
      auto it = info.vecBaseMap.find(meas);
      if (it == info.vecBaseMap.end()) {
        op.emitOpError("measurements collection operand #")
            << i << " could not be resolved. Ensure it originates from "
            << "a quake.mz/mx/my operation on a qubit register.";
        return failure();
      }
      if (!measTy.hasSpecifiedSize()) {
        op.emitOpError("measurements collection operand #")
            << i << " has dynamic size (!quake.measurements<?>). "
            << "QEC lowering requires statically-known sizes.";
        return failure();
      }
      auto base = it->second;
      for (std::size_t j = 0; j < measTy.getSize(); j++)
        indices.push_back(base + static_cast<std::int64_t>(j));
    }
  }

  if (indices.empty()) {
    op.erase();
    return success();
  }

  auto count = static_cast<std::int64_t>(indices.size());
  auto countVal = builder.create<arith::ConstantIntOp>(loc, count, 64);
  Value arrayAlloc = builder.create<cudaq::cc::AllocaOp>(loc, i64Ty, countVal);

  for (auto [i, idx] : llvm::enumerate(indices)) {
    auto idxVal = builder.create<arith::ConstantIntOp>(loc, idx, 64);
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

/// Lower all QEC ops within a single function. Uses the function's own
/// measurement index map and total measurement count -- NOT a module-wide
/// aggregate. This ensures correct rec[-N] lookback computation even when
/// multiple kernel functions with different measurement counts coexist in
/// the same module.
static bool lowerQECOpsInFunction(func::FuncOp func, ModuleOp module) {
  auto info = buildMeasurementIndexMapForFunc(func);
  if (info.hadError)
    return true;

  SmallVector<qec::DetectorOp> detectorOps;
  SmallVector<qec::LogicalObservableOp> observableOps;
  SmallVector<qec::DetectorsVectorizedOp> vectorizedOps;
  func.walk([&](qec::DetectorOp op) { detectorOps.push_back(op); });
  func.walk(
      [&](qec::LogicalObservableOp op) { observableOps.push_back(op); });
  func.walk(
      [&](qec::DetectorsVectorizedOp op) { vectorizedOps.push_back(op); });

  // No QEC ops in this function -- nothing to do.
  if (detectorOps.empty() && observableOps.empty() && vectorizedOps.empty())
    return false;

  bool failed = false;

  for (auto op : detectorOps) {
    OpBuilder builder(op);
    auto totalVal = builder.create<arith::ConstantIntOp>(
        op.getLoc(), info.totalMeasurements, 64);
    if (lowerQECOpToCall(op, info, module,
                         cudaq::opt::QIRDetectorIndices,
                         SmallVector<Value>{totalVal})
            .failed())
      failed = true;
  }

  for (auto op : observableOps) {
    OpBuilder builder(op);
    auto loc = op.getLoc();
    auto totalVal =
        builder.create<arith::ConstantIntOp>(loc, info.totalMeasurements, 64);
    auto obsIdx = op.getObservableIndex();
    auto obsIdxVal = builder.create<arith::ConstantIntOp>(loc, obsIdx, 64);
    if (lowerQECOpToCall(op, info, module,
                         cudaq::opt::QIRLogicalObservableIndices,
                         SmallVector<Value>{totalVal, obsIdxVal})
            .failed())
      failed = true;
  }

  // detectors_vectorized: both operands are !quake.measurements<N>
  // originating from multi-qubit mz(). Resolve each to its base measurement
  // index, then emit a call to __quantum__qis__detectors_vectorized_indices.
  // This enables cross-round detectors (the fundamental QEC pattern) to
  // work in compiled mode without manually unrolling to N detector() calls.
  for (auto op : vectorizedOps) {
    auto prevVec = op.getPrev();
    auto currVec = op.getCurr();

    auto prevIt = info.vecBaseMap.find(prevVec);
    auto currIt = info.vecBaseMap.find(currVec);
    if (prevIt == info.vecBaseMap.end() ||
        currIt == info.vecBaseMap.end()) {
      op.emitOpError("detectors_vectorized operands must originate from "
                     "multi-qubit measurements (quake.mz on veq). Could not "
                     "resolve measurement collection base index.");
      failed = true;
      continue;
    }

    // Get element counts directly from the MeasurementsType.
    auto prevMeasTy = cast<quake::MeasurementsType>(prevVec.getType());
    auto currMeasTy = cast<quake::MeasurementsType>(currVec.getType());
    if (!prevMeasTy.hasSpecifiedSize() || !currMeasTy.hasSpecifiedSize()) {
      op.emitOpError("detectors_vectorized requires statically-sized "
                     "!quake.measurements<N> operands, got unsized <?>.");
      failed = true;
      continue;
    }
    auto prevCount = static_cast<std::int64_t>(prevMeasTy.getSize());
    auto currCount = static_cast<std::int64_t>(currMeasTy.getSize());
    if (prevCount != currCount) {
      op.emitOpError("detectors_vectorized requires both collections to "
                     "have equal sizes. Got prev=")
          << prevCount << ", curr=" << currCount;
      failed = true;
      continue;
    }

    OpBuilder builder(op);
    auto loc = op.getLoc();
    auto i64Ty = builder.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto countVal =
        builder.create<arith::ConstantIntOp>(loc, prevCount, 64);
    auto totalVal = builder.create<arith::ConstantIntOp>(
        loc, info.totalMeasurements, 64);

    // Allocate and fill prev index array
    Value prevAlloc =
        builder.create<cudaq::cc::AllocaOp>(loc, i64Ty, countVal);
    for (std::int64_t i = 0; i < prevCount; i++) {
      auto idxVal = builder.create<arith::ConstantIntOp>(
          loc, prevIt->second + i, 64);
      auto elemPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI64Ty, prevAlloc,
          ArrayRef<cudaq::cc::ComputePtrArg>{static_cast<int32_t>(i)});
      builder.create<cudaq::cc::StoreOp>(loc, idxVal, elemPtr);
    }

    // Allocate and fill curr index array
    Value currAlloc =
        builder.create<cudaq::cc::AllocaOp>(loc, i64Ty, countVal);
    for (std::int64_t i = 0; i < prevCount; i++) {
      auto idxVal = builder.create<arith::ConstantIntOp>(
          loc, currIt->second + i, 64);
      auto elemPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI64Ty, currAlloc,
          ArrayRef<cudaq::cc::ComputePtrArg>{static_cast<int32_t>(i)});
      builder.create<cudaq::cc::StoreOp>(loc, idxVal, elemPtr);
    }

    Value prevCast =
        builder.create<cudaq::cc::CastOp>(loc, ptrI64Ty, prevAlloc);
    Value currCast =
        builder.create<cudaq::cc::CastOp>(loc, ptrI64Ty, currAlloc);

    SmallVector<Value> callArgs = {prevCast, currCast, countVal, totalVal};
    SmallVector<Type> argTypes = {ptrI64Ty, ptrI64Ty, i64Ty, i64Ty};
    auto funcTy =
        FunctionType::get(builder.getContext(), argTypes, /*results=*/{});
    auto funcDecl = cudaq::opt::factory::createFunction(
        cudaq::opt::QIRDetectorsVectorizedIndices, funcTy.getResults(),
        funcTy.getInputs(), module);
    funcDecl.setPrivate();
    builder.create<func::CallOp>(
        loc, TypeRange{}, cudaq::opt::QIRDetectorsVectorizedIndices, callArgs);
    op.erase();
  }

  return failed;
}

class LowerQECToQIRPass
    : public cudaq::opt::impl::LowerQECToQIRBase<LowerQECToQIRPass> {
public:
  using LowerQECToQIRBase::LowerQECToQIRBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto module = dyn_cast<ModuleOp>(moduleOp);
    LLVM_DEBUG(llvm::dbgs() << "Before QEC lowering\n");

    bool failed = false;
    // Process each function independently. Measurement indices and
    // totalMeasurements are per-function, matching QIR execution semantics.
    moduleOp->walk([&](func::FuncOp func) {
      if (lowerQECOpsInFunction(func, module))
        failed = true;
    });

    if (failed)
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "After QEC lowering\n");
  }
};
} // namespace
