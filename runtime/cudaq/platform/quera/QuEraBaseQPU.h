/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The `QuEraBaseQPU` is a QPU that allows users to
// submit kernels to the QuEra machine.
class QuEraBaseQPU : public BaseRemoteRESTQPU {
protected:
  std::tuple<mlir::ModuleOp, mlir::MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    throw std::runtime_error("Not supported on this target.");
  }

public:
  virtual bool isRemote() override { return true; }

  virtual bool isEmulated() override { return false; }

  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    throw std::runtime_error("Not supported on this target.");
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    if ("QuEra_Analog_Hamiltoninan" != kernelName) {
      throw std::runtime_error("Not supported on this target.");
    }
  }
};
} // namespace cudaq
