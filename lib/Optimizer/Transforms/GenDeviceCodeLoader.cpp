/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CallGraphFix.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GENERATEDEVICECODELOADER
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "device-code-loader"

using namespace mlir;

namespace {

/// @brief Return the size and member variable offsets for the kernel's return
/// type.
std::pair<std::size_t, std::vector<std::size_t>> inline extractReturnLayout(
    mlir::func::FuncOp kernelFunc) {
  std::size_t totalSize = 0;
  std::vector<std::size_t> fieldOffsets;
  // Only proceed if function has a return type
  if (kernelFunc.getNumResults() > 0) {
    mlir::Type returnType = kernelFunc.getResultTypes()[0];
    auto mod = kernelFunc->getParentOfType<ModuleOp>();
    StringRef dataLayoutSpec = "";
    if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    auto dataLayout = llvm::DataLayout(dataLayoutSpec);
    // Convert bufferTy to llvm.
    llvm::LLVMContext context;
    LLVMTypeConverter converter(kernelFunc.getContext());
    cudaq::opt::initializeTypeConversions(converter);
    // Handle structure types
    if (auto structType = mlir::dyn_cast<cudaq::cc::StructType>(returnType)) {
      auto llvmDialectTy = converter.convertType(structType);
      LLVM::TypeToLLVMIRTranslator translator(context);
      auto *llvmStructTy =
          cast<llvm::StructType>(translator.translateType(llvmDialectTy));
      auto *layout = dataLayout.getStructLayout(llvmStructTy);
      totalSize = layout->getSizeInBytes();
      std::vector<std::size_t> fieldOffsets;
      std::size_t numElements = structType.getMembers().size();
      for (std::size_t i = 0; i < numElements; ++i)
        fieldOffsets.emplace_back(layout->getElementOffset(i));
    }
    /// Temporarily commented out ld.lld: error: undefined symbol:
    /// cudaq::opt::getDataSize(llvm::DataLayout&, mlir::Type)
    // else
    //   totalSize = cudaq::opt::getDataSize(dataLayout, returnType);
  }
  return {totalSize, fieldOffsets};
}

class GenerateDeviceCodeLoaderPass
    : public cudaq::opt::impl::GenerateDeviceCodeLoaderBase<
          GenerateDeviceCodeLoaderPass> {
public:
  using GenerateDeviceCodeLoaderBase::GenerateDeviceCodeLoaderBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    assert(ctx && "must have a context");
    auto loc = module->getLoc();
    std::error_code ec;
    llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Failed to open output file '" << outputFilename << "'\n";
      std::exit(ec.value());
    }
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    if (generateAsQuake) {
      // Add declaration of deviceCodeHolderAdd
      builder.create<LLVM::LLVMFuncOp>(
          loc, cudaq::runtime::deviceCodeHolderAdd,
          LLVM::LLVMFunctionType::get(
              cudaq::opt::factory::getVoidType(ctx),
              {cudaq::opt::factory::getPointerType(ctx),
               cudaq::opt::factory::getPointerType(ctx)}));
    }

    auto mangledNameMap =
        module->getAttrOfType<DictionaryAttr>(cudaq::runtime::mangledNameMap);

    // Collect all function declarations to forward as part of each Module.
    // These are thrown in so the Module's CallOps are complete. Unused
    // declarations are just thrown away when the code is JIT compiled.
    // Also look for any global symbols associated with custom operations
    SmallVector<Operation *> declarations;
    for (auto &op : *module.getBody()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "adding declaration: " << op);
          declarations.push_back(&op);
        }
      } else if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        if (funcOp.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "adding declaration: " << op);
          declarations.push_back(&op);
        }
      } else if (auto ccGlobalOp = dyn_cast<cudaq::cc::GlobalOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding global constants: " << op);
        declarations.push_back(&op);
      }
    }

    // Create a call graph to track kernel dependency.
    mlir::CallGraph callGraph(module);
    for (auto &op : *module.getBody()) {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp)
        continue;
      if (!funcOp.getName().startswith(cudaq::runtime::cudaqGenPrefixName))
        continue;
      if (funcOp->hasAttr(cudaq::generatorAnnotation) || funcOp.empty())
        continue;
      auto className =
          funcOp.getName().drop_front(cudaq::runtime::cudaqGenPrefixLength);
      LLVM_DEBUG(llvm::dbgs() << "processing function " << className << '\n');
      // Generate LLVM-IR dialect to register the device code loading.
      std::string thunkName = className.str() + ".thunk";
      std::string funcCode;
      llvm::raw_string_ostream strOut(funcCode);
      OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      strOut << "module attributes " << module->getAttrDictionary() << " { ";

      // We'll also need any non-inlined functions that are
      // called by our cudaq kernel
      // Set of dependent kernels that we've included.
      // Note: the `CallGraphNode` does include 'this' function.
      mlir::CallGraphNode *node =
          callGraph.lookupNode(funcOp.getCallableRegion());
      // Iterate over all dependent kernels starting at this node.
      for (auto it = llvm::df_begin(node), itEnd = llvm::df_end(node);
           it != itEnd; ++it) {
        // Only consider those that are defined in this module.
        if (!it->isExternal()) {
          auto *callableRegion = it->getCallableRegion();
          auto parentFuncOp =
              callableRegion->getParentOfType<mlir::func::FuncOp>();
          LLVM_DEBUG(llvm::dbgs() << "  Adding dependent function "
                                  << parentFuncOp->getName() << '\n');
          parentFuncOp.print(strOut, opf);
          strOut << '\n';
        }
      }

      // Include the generated kernel thunk if present since it is on the
      // callee side of the launchKernel() callback.
      if (auto *thunkFunc = module.lookupSymbol(thunkName)) {
        LLVM_DEBUG(llvm::dbgs() << "found thunk function\n");
        strOut << *thunkFunc << '\n';
      }
      if (auto *zeroDynRes = module.lookupSymbol("__nvqpp_zeroDynamicResult")) {
        LLVM_DEBUG(llvm::dbgs() << "found zero dyn result function\n");
        strOut << *zeroDynRes << '\n';
      }
      if (auto *createDynRes =
              module.lookupSymbol("__nvqpp_createDynamicResult")) {
        LLVM_DEBUG(llvm::dbgs() << "found create dyn result function\n");
        strOut << *createDynRes << '\n';
      }

      // Conservatively, include all declarations. (Unreferenced ones can be
      // erased with a symbol DCE.)
      for (auto *op : declarations)
        strOut << *op << '\n';
      strOut << "\n}\n" << '\0';

      auto devCode = builder.create<LLVM::GlobalOp>(
          loc, cudaq::opt::factory::getStringType(ctx, funcCode.size()),
          /*isConstant=*/true, LLVM::Linkage::Private,
          className.str() + "CodeHolder.extract_device_code",
          builder.getStringAttr(funcCode), /*alignment=*/0);
      auto devName = builder.create<LLVM::GlobalOp>(
          loc, cudaq::opt::factory::getStringType(ctx, className.size() + 1),
          /*isConstant=*/true, LLVM::Linkage::Private,
          className.str() + "CodeHolder.extract_device_name",
          builder.getStringAttr(className.str() + '\0'), /*alignment=*/0);
      auto initFun = builder.create<LLVM::LLVMFuncOp>(
          loc, className.str() + ".init_func",
          LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx),
                                      {}));
      auto insPt = builder.saveInsertionPoint();
      auto *initFunEntry = initFun.addEntryBlock();
      builder.setInsertionPointToStart(initFunEntry);
      auto devRef = builder.create<LLVM::AddressOfOp>(
          loc, cudaq::opt::factory::getPointerType(devName.getType()),
          devName.getSymName());
      auto codeRef = builder.create<LLVM::AddressOfOp>(
          loc, cudaq::opt::factory::getPointerType(devCode.getType()),
          devCode.getSymName());
      auto castDevRef = builder.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(ctx), devRef);
      auto castCodeRef = builder.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(ctx), codeRef);
      builder.create<LLVM::CallOp>(loc, std::nullopt,
                                   cudaq::runtime::deviceCodeHolderAdd,
                                   ValueRange{castDevRef, castCodeRef});

      auto kernName = funcOp.getSymName().str();
      if (!jitTime && mangledNameMap && !mangledNameMap.empty() &&
          mangledNameMap.contains(kernName)) {
        auto hostFuncNameAttr = mangledNameMap.getAs<StringAttr>(kernName);
        auto hostFuncName = hostFuncNameAttr.getValue();
        auto hostFuncOp = module.lookupSymbol<func::FuncOp>(hostFuncName);
        if (!hostFuncOp) {
          // Using a fake type. We just want the symbol of an artifact defined
          // in host code. We're not calling this function.
          hostFuncOp =
              cudaq::opt::factory::createFunction(hostFuncName, {}, {}, module);
          hostFuncOp.setPrivate();
        }
        auto ptrTy = cudaq::cc::PointerType::get(builder.getI8Type());
        auto entryRef = builder.create<func::ConstantOp>(
            loc, hostFuncOp.getFunctionType(), hostFuncOp.getSymName());
        auto castEntryRef =
            builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrTy, entryRef);
        auto deviceRef = builder.create<func::ConstantOp>(
            loc, funcOp.getFunctionType(), funcOp.getSymName());
        auto castDeviceRef =
            builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrTy, deviceRef);
        auto castKernNameRef =
            builder.create<cudaq::cc::CastOp>(loc, ptrTy, devRef);

        cudaq::IRBuilder irBuilder(builder);
        if (failed(irBuilder.loadIntrinsic(
                module, cudaq::runtime::registerLinkableKernel))) {
          signalPassFailure();
        }
        builder.create<func::CallOp>(
            loc, std::nullopt, cudaq::runtime::registerLinkableKernel,
            ValueRange{castEntryRef, castKernNameRef, castDeviceRef});

        /// ASKME: Save for all return types or only struct types is sufficient?
        if (funcOp.getNumResults() > 0 &&
            isa<cudaq::cc::StructType>(funcOp.getResultTypes()[0])) {
          auto [totalSize, fieldOffsets] = extractReturnLayout(funcOp);
          auto i64Ty = builder.getI64Type();
          auto ptrTy = cudaq::cc::PointerType::get(builder.getI8Type());
          // Create global constant with the total size
          std::string sizeName = className.str() + ".returnTy.struct_size";
          auto sizeGlobal = builder.create<LLVM::GlobalOp>(
              loc, i64Ty, /*isConstant=*/true, LLVM::Linkage::Private, sizeName,
              builder.getI64IntegerAttr(totalSize), /*alignment=*/0);
          auto sizeRef = builder.create<LLVM::AddressOfOp>(
              loc, cudaq::opt::factory::getPointerType(sizeGlobal.getType()),
              sizeGlobal.getSymName());
          auto castSizeRef =
              builder.create<cudaq::cc::CastOp>(loc, ptrTy, sizeRef);
          // Create global constant with the field offsets
          auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, fieldOffsets.size());
          SmallVector<mlir::Attribute> offsetAttrs;
          for (auto offset : fieldOffsets)
            offsetAttrs.push_back(builder.getI64IntegerAttr(offset));
          auto offsetsArrayAttr = builder.getArrayAttr(offsetAttrs);
          std::string offsetsName = className.str() + ".returnTy.field_offsets";
          auto offsetsGlobal = builder.create<LLVM::GlobalOp>(
              loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Private,
              offsetsName, builder.getArrayAttr(offsetsArrayAttr),
              /*alignment=*/0);
          auto offsetsRef = builder.create<LLVM::AddressOfOp>(
              loc, cudaq::opt::factory::getPointerType(offsetsGlobal.getType()),
              offsetsGlobal.getSymName());
          auto castOffsetsRef =
              builder.create<cudaq::cc::CastOp>(loc, ptrTy, offsetsRef);
          // Create global constant with the number of offsets
          std::string numOffsetsName =
              className.str() + ".returnTy.num_offsets";
          auto numOffsetsGlobal = builder.create<LLVM::GlobalOp>(
              loc, i64Ty, /*isConstant=*/true, LLVM::Linkage::Private,
              numOffsetsName, builder.getI64IntegerAttr(fieldOffsets.size()),
              /*alignment=*/0);
          auto numOffsetsRef = builder.create<LLVM::AddressOfOp>(
              loc,
              cudaq::opt::factory::getPointerType(numOffsetsGlobal.getType()),
              numOffsetsGlobal.getSymName());
          auto castNumOffsetsRef = builder.create<cudaq::cc::CastOp>(
              loc, cudaq::cc::PointerType::get(builder.getI8Type()),
              numOffsetsRef);
          builder.create<func::CallOp>(
              loc, std::nullopt, cudaq::runtime::returnTypeLayoutAdd,
              ValueRange{castKernNameRef, castSizeRef, castOffsetsRef,
                         castNumOffsetsRef});
        }
      }

      builder.create<LLVM::ReturnOp>(loc, ValueRange{});
      builder.restoreInsertionPoint(insPt);
      cudaq::opt::factory::createGlobalCtorCall(
          module, mlir::FlatSymbolRefAttr::get(ctx, initFun.getName()));
    }
    out.keep();
  }
};
} // namespace
