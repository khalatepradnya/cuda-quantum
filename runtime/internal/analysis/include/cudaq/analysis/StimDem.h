/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file StimDem.h
/// @brief Option C surface: thin recorder primitive returning the structured
///        Detector Error Model as a `stim::DetectorErrorModel` value.
///
/// This is the cross-repo entry point intended to be consumed by CUDA-QX's
/// `cudaq::qec::dem_from_kernel(...)` user-facing wrapper. It deliberately
/// avoids the text round-trip that `cudaq::analysis::computeDem` currently
/// uses (Option A); the structured DEM crosses the library boundary by value
/// instead. With C++17 mandatory copy elision and Stim's vector-backed DEM
/// type having `noexcept` move semantics, the API boundary cost is zero
/// regardless of DEM size.
///
/// Trade-off: callers must build/link against a Stim version compatible with
/// the one CUDA-Q vendors in `tpls/Stim/`, because `stim::DetectorErrorModel`
/// is exchanged across the boundary. This is the same managed-coupling we
/// already accept for MLIR/LLVM types between CUDA-Q and CUDA-QX.

#pragma once

#include "cudaq/analysis/Dem.h" // for ScopedAnalysisSimulator

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"

#include "stim.h"

#include <concepts>
#include <functional>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
} // namespace cudaq

namespace cudaq::analysis {

namespace detail {

/// @brief Type-erased core of `record_dem`. Lives in Dem.cpp so the templated
///        entry point in this header does not need to drag the
///        `quantum_platform` machinery into every translation unit that
///        consumes the API.
stim::DetectorErrorModel
runRecordDem(const std::string &kernelName,
             cudaq::quantum_platform &platform,
             const cudaq::noise_model *noise,
             const std::function<void()> &wrappedKernel);

} // namespace detail

/// @brief Run the DEM analysis engine over a CUDA-Q kernel and return the
///        resulting `stim::DetectorErrorModel` directly, with no textual
///        serialisation between the recorder and the caller.
///
/// Same kernel-side semantics as `computeDem`: the kernel is executed through
/// the Stim simulator (forced via `ScopedAnalysisSimulator`) under an
/// `ExecutionContext` named "dem"; the simulator records the circuit
/// including `DETECTOR` and `OBSERVABLE_INCLUDE` instructions emitted by the
/// `qec.*` annotation ops; `stim::ErrorAnalyzer::circuit_to_detector_error_model`
/// constructs the DEM from that recorded circuit.
///
/// Differs from `computeDem` only in the return type: the structured
/// `stim::DetectorErrorModel` instead of a `DemData` carrying its textual
/// serialisation.
///
/// @param kernel  A CUDA-Q kernel (any callable invocable with @p args).
/// @param noise   Optional noise model to attach for the analysis run.
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
stim::DetectorErrorModel record_dem(QuantumKernel &&kernel,
                                    const cudaq::noise_model *noise,
                                    Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return detail::runRecordDem(kernelName, platform, noise, [&]() mutable {
    kernel(std::forward<Args>(args)...);
  });
}

/// @brief Convenience overload for the no-noise case.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
stim::DetectorErrorModel record_dem(QuantumKernel &&kernel, Args &&...args) {
  return record_dem(std::forward<QuantumKernel>(kernel),
                    /*noise=*/nullptr, std::forward<Args>(args)...);
}

} // namespace cudaq::analysis
