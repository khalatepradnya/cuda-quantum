/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Tools/ParseUtilities.h"

using namespace mlir;

namespace cudaq {
static bool mlirLLVMInitialized = false;

static llvm::StringMap<cudaq::Translation> &getTranslationRegistry() {
  static llvm::StringMap<cudaq::Translation> translationBundle;
  return translationBundle;
}
cudaq::Translation &getTranslation(StringRef name) {
  auto &registry = getTranslationRegistry();
  if (!registry.count(name))
    throw std::runtime_error("Invalid IR Translation (" + name.str() + ").");
  return registry[name];
}

static void registerTranslation(StringRef name, StringRef description,
                                const TranslateFromMLIRFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(function, description);
}

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function) {
  registerTranslation(name, description, function);
}
} // namespace cudaq

#include "RuntimeMLIRCommonImpl.h"

namespace cudaq {

std::unique_ptr<MLIRContext> initializeMLIR() {
  if (!mlirLLVMInitialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    cudaq::registerAllPasses();
    registerToQIRTranslation();
    registerToOpenQASMTranslation();
    registerToIQMJsonTranslation();
    mlirLLVMInitialized = true;
  }

  DialectRegistry registry;
  cudaq::opt::registerCodeGenDialect(registry);
  cudaq::registerAllDialects(registry);
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  registerLLVMDialectTranslation(*context);
  return context;
}

/// Extract data layout information from kernel's return type
std::pair<std::size_t, std::vector<std::size_t>>
extractDataLayout(const std::string &kernelName) {
  auto mlirContext = initializeMLIR();
  auto quakeCode = cudaq::get_quake_by_name(kernelName);
  auto m_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(quakeCode), mlirContext.get());
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");
  mlir::func::FuncOp kernelFunc;
  m_module->walk([&](mlir::func::FuncOp fOp) {
    if (fOp.getName().equals("__nvqpp__mlirgen__" + kernelName)) {
      kernelFunc = fOp;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in the module.");
  // Extract layout information from the function's return type
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
    llvm::LLVMContext context;
    LLVMTypeConverter converter(kernelFunc.getContext());
    cudaq::opt::initializeTypeConversions(converter);
    // Handle structure types
    if (auto structType = mlir::dyn_cast<cc::StructType>(returnType)) {
      auto llvmDialectTy = converter.convertType(structType);
      LLVM::TypeToLLVMIRTranslator translator(context);
      auto *llvmStructTy =
          cast<llvm::StructType>(translator.translateType(llvmDialectTy));
      auto *layout = dataLayout.getStructLayout(llvmStructTy);
      totalSize = layout->getSizeInBytes();
      std::vector<std::size_t> fieldOffsets;
      for (std::size_t i = 0, I = structType.getMembers().size(); i != I; ++i)
        fieldOffsets.emplace_back(layout->getElementOffset(i));
    } else {
      // For non-struct types, just the size
      totalSize = cudaq::opt::getDataSize(dataLayout, returnType);
    }
  }
  return {totalSize, fieldOffsets};
}

} // namespace cudaq
