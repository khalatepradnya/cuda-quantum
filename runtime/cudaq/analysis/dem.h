/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform.h"
#include "nvqir/AnalysisScope.h"

#include "stim.h"

#include <concepts>
#include <functional>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
} // namespace cudaq

namespace cudaq::analysis {

namespace dem {

/// @brief Activate the DEM (Detector Error Model) analysis on the current
/// thread, backed by an NVQIR simulator plugin.
///
/// The returned `nvqir::AnalysisScope` claims the thread-local analysis slot,
/// resets the backend's recorded circuit on entry, and releases the slot on
/// destruction. While the scope is alive, every gate, measurement, noise
/// channel, and `qec.*` annotation lowered out of the kernel is appended to
/// the simulator's recorded circuit. `compute_dem` consumes the recorded
/// circuit after the kernel returns and feeds it to Stim's `ErrorAnalyzer`.
///
/// @param plugin_name  NVQIR simulator plugin to drive the analysis. Defaults
///                     to "stim", which is the only backend that produces a
///                     non-null `getRecordedCircuit()` today. Custom plugins
///                     may be substituted as long as they implement the
///                     `detector` / `observable` / recorded-circuit
///                     virtuals on `nvqir::CircuitSimulator`.
///
/// Throws `std::runtime_error` if an analysis scope is already active on the
/// current thread, or if the plugin cannot be resolved.
nvqir::AnalysisScope make_scope(std::string plugin_name = "stim");

} // namespace dem

namespace detail {

/// @brief Type-erased core of `compute_dem`. Lives in the analysis library
/// so the templated entry point in this header does not pull
/// `quantum_platform`, the kernel builder, or Stim's internals into every
/// translation unit that consumes the API.
stim::DetectorErrorModel
runComputeDem(const std::string &kernelName, cudaq::quantum_platform &platform,
              const cudaq::noise_model *noise,
              const std::function<void()> &wrappedKernel,
              std::string plugin_name = "stim");

} // namespace detail

/// @brief Run the DEM analysis over a CUDA-Q kernel and return the resulting
/// `stim::DetectorErrorModel` by value.
///
/// The kernel is executed under an `ExecutionContext` named "dem", with the
/// thread-local analysis simulator forced via `dem::make_scope`. The simulator
/// records the full circuit including `DETECTOR` and `OBSERVABLE_INCLUDE`
/// instructions lowered from `qec.*` ops; after the kernel returns,
/// `stim::ErrorAnalyzer::circuit_to_detector_error_model` builds the DEM from
/// that recorded circuit. Returning the structured DEM by value (RVO + Stim's
/// noexcept move) avoids the text round-trip an `std::string` API would
/// impose.
///
/// @note The active CUDA-Q target / platform is not modified; the analysis
///       simulator is purely an internal override.
///
/// @param kernel  Any callable invocable with @p args (CUDA-Q kernel, lambda,
///                or kernel-builder).
/// @param noise   Optional noise model layered on per `cudaq::noise_model`
///                semantics. When null, the kernel's `apply_noise` ops (if
///                any) are the sole noise source.
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
stim::DetectorErrorModel compute_dem(QuantumKernel &&kernel,
                                     const cudaq::noise_model *noise,
                                     Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return detail::runComputeDem(kernelName, platform, noise, [&]() mutable {
    kernel(std::forward<Args>(args)...);
  });
}

/// @brief Convenience overload for the no-noise case.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
stim::DetectorErrorModel compute_dem(QuantumKernel &&kernel, Args &&...args) {
  return compute_dem(std::forward<QuantumKernel>(kernel),
                     /*noise=*/nullptr, std::forward<Args>(args)...);
}

} // namespace cudaq::analysis
