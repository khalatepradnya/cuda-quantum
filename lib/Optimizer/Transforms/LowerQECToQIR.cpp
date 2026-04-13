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
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERQECTOQIR
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lower-qec-to-qir"

using namespace mlir;

namespace {

/// Check whether \p ty is a pointer to a QIR "Array" struct, i.e. the
/// converted form of !quake.measurements<N>.
static bool isArrayPtrType(Type ty) {
  if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty)) {
    if (auto structTy =
            dyn_cast<LLVM::LLVMStructType>(ptrTy.getElementType()))
      return structTy.getName() == "Array";
    // Opaque pointer case
    if (isa<NoneType>(ptrTy.getElementType()))
      return false; // ambiguous, treat as scalar
  }
  return false;
}

/// Emit a func.call to \p funcName with argument types \p argTypes. Declares
/// the function in \p module if it doesn't already exist.
static void emitCallToRuntimeFunc(OpBuilder &builder, Location loc,
                                  ModuleOp module, StringRef funcName,
                                  SmallVector<Value> &args,
                                  SmallVector<Type> &argTypes) {
  auto funcTy =
      FunctionType::get(builder.getContext(), argTypes, /*results=*/{});
  auto funcDecl = cudaq::opt::factory::createFunction(
      funcName, funcTy.getResults(), funcTy.getInputs(), module);
  funcDecl.setPrivate();
  builder.create<func::CallOp>(loc, TypeRange{}, funcName, args);
}

/// Lower a detector or logical_observable op whose operands are post-QIR
/// converted values (Result* pointers or Array* pointers from measurement
/// collections).
///
/// Two cases:
///  1. All operands are scalar Result* → pack into a stack-allocated
///     Result*[] array and call the _from_results runtime function.
///  2. A single Array* operand → call the _from_array runtime function.
template <typename QECOp>
static void lowerQECOp(QECOp op, ModuleOp module, StringRef resultsFunc,
                       StringRef arrayFunc,
                       SmallVector<Value> extraArgs = {}) {
  OpBuilder builder(op);
  auto loc = op.getLoc();
  auto i64Ty = builder.getI64Type();
  auto operands = op.getMeasurements();

  if (operands.empty()) {
    op.erase();
    return;
  }

  // Case: single Array* operand (converted !quake.measurements<N>)
  if (operands.size() == 1 && isArrayPtrType(operands[0].getType())) {
    SmallVector<Type> argTypes = {operands[0].getType()};
    SmallVector<Value> args = {operands[0]};
    for (auto v : extraArgs) {
      argTypes.push_back(v.getType());
      args.push_back(v);
    }
    emitCallToRuntimeFunc(builder, loc, module, arrayFunc, args, argTypes);
    op.erase();
    return;
  }

  // Case: individual Result* operands → pack into Result*[] stack array.
  auto count = static_cast<std::int64_t>(operands.size());
  auto countVal = builder.create<arith::ConstantIntOp>(loc, count, 64);

  // Result* is a pointer type; we store pointers, so element type is Result*.
  auto resultPtrTy = operands[0].getType();
  auto ptrToPtrTy = cudaq::cc::PointerType::get(resultPtrTy);
  Value arrayAlloc =
      builder.create<cudaq::cc::AllocaOp>(loc, resultPtrTy, countVal);

  for (auto [i, meas] : llvm::enumerate(operands)) {
    auto constIdx = static_cast<std::int32_t>(i);
    auto elemPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrToPtrTy, arrayAlloc,
        ArrayRef<cudaq::cc::ComputePtrArg>{constIdx});
    builder.create<cudaq::cc::StoreOp>(loc, meas, elemPtr);
  }

  Value castPtr =
      builder.create<cudaq::cc::CastOp>(loc, ptrToPtrTy, arrayAlloc);
  SmallVector<Value> args = {castPtr, countVal};
  SmallVector<Type> argTypes = {ptrToPtrTy, i64Ty};
  for (auto v : extraArgs) {
    argTypes.push_back(v.getType());
    args.push_back(v);
  }

  emitCallToRuntimeFunc(builder, loc, module, resultsFunc, args, argTypes);
  op.erase();
}

class LowerQECToQIRPass
    : public cudaq::opt::impl::LowerQECToQIRBase<LowerQECToQIRPass> {
public:
  using LowerQECToQIRBase::LowerQECToQIRBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto module = dyn_cast<ModuleOp>(moduleOp);
    LLVM_DEBUG(llvm::dbgs() << "LowerQECToQIR: before lowering\n");

    SmallVector<qec::DetectorOp> detectorOps;
    SmallVector<qec::LogicalObservableOp> observableOps;
    SmallVector<qec::DetectorsVectorizedOp> vectorizedOps;
    moduleOp->walk([&](qec::DetectorOp op) { detectorOps.push_back(op); });
    moduleOp->walk(
        [&](qec::LogicalObservableOp op) { observableOps.push_back(op); });
    moduleOp->walk(
        [&](qec::DetectorsVectorizedOp op) { vectorizedOps.push_back(op); });

    if (detectorOps.empty() && observableOps.empty() && vectorizedOps.empty())
      return;

    for (auto op : detectorOps)
      lowerQECOp(op, module, cudaq::opt::QIRDetectorFromResults,
                 cudaq::opt::QIRDetectorFromArray);

    for (auto op : observableOps) {
      OpBuilder builder(op);
      auto obsIdxVal = builder.create<arith::ConstantIntOp>(
          op.getLoc(), op.getObservableIndex(), 64);
      lowerQECOp(op, module, cudaq::opt::QIRLogicalObservableFromResults,
                 cudaq::opt::QIRLogicalObservableFromArray,
                 SmallVector<Value>{obsIdxVal});
    }

    for (auto op : vectorizedOps) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      auto prev = op.getPrev();
      auto curr = op.getCurr();

      // Both operands should be Array* (from converted !quake.measurements<N>)
      SmallVector<Type> argTypes = {prev.getType(), curr.getType()};
      SmallVector<Value> args = {prev, curr};
      emitCallToRuntimeFunc(
          builder, loc, module,
          cudaq::opt::QIRDetectorsVectorizedFromArrays, args, argTypes);
      op.erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "LowerQECToQIR: after lowering\n");
  }
};
} // namespace
