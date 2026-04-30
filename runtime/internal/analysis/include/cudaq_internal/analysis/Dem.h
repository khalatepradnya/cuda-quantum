/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file Dem.h
/// @brief Internal Public Module API for the DEM (Detector Error Model)
/// analysis engine.
///
/// This header is part of the `cudaq_internal::analysis` module per
/// `CppAPICodingStyle.md` §3.2 (Internal Public Module APIs). It is meant to
/// be consumed by CUDA-QX's `cudaq::qec::dem_from_kernel(...)` user-facing
/// wrapper, not by end users directly.
///
/// Concepts:
///
///   - `dem_policy`     Tag + options struct for DEM analysis. Mirrors the
///                      shape of `cudaq::sample_policy` /
///                      `cudaq::observe_policy` so dispatch through the
///                      existing CPO infrastructure
///                      (`policy_dispatch.h`, `policy_cpos.h`) works without
///                      bespoke plumbing. Defined in `DemPolicy.h`.
///
///   - `DemData`        Result type carried by `dem_policy::result_type`.
///                      Defined in `DemPolicy.h`.
///
///   - `computeDem`     Templated entry point that drives a kernel through
///                      the DEM analysis path. Parallels `estimate_resources`
///                      in shape — wraps kernel + args, sets up the analysis
///                      execution context, runs through
///                      `with_execution_context`, returns `DemData`.
///
/// "Stim" is an implementation detail of the DEM engine, not part of its
/// public surface. The engine may be re-implemented over a different backend
/// (direct IR walker, alternative Pauli-frame analyzer, etc.) without
/// changing this header.

#pragma once

#include "cudaq_internal/analysis/DemPolicy.h"

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"

#include <concepts>
#include <functional>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
} // namespace cudaq

namespace cudaq_internal::analysis {

namespace detail {

/// @brief Force the NVQIR thread-local simulator pointer to the Stim plugin
///        for the duration of the DEM analysis run, regardless of the active
///        target.
///
/// FIXME(runtime-team): This is the second of two gaps that make DEM the
/// architecturally-awkward third user of the policy infrastructure. The
/// simulator selection should be a CPO derived from the policy
/// (`select_simulator_for_policy`?) so each analysis engine declares its
/// simulator dependency in its own policy struct, rather than `computeDem`
/// hard-coding "stim". Until that lands, this RAII guard mirrors the existing
/// `nvqir::switchToResourceCounterSimulator` pattern in shape (thread-local
/// flag + restore-on-destruct) but with explicit RAII semantics so exceptions
/// do not leak the override.
class ScopedAnalysisSimulator {
public:
  ScopedAnalysisSimulator();
  ~ScopedAnalysisSimulator();
  ScopedAnalysisSimulator(const ScopedAnalysisSimulator &) = delete;
  ScopedAnalysisSimulator &operator=(const ScopedAnalysisSimulator &) = delete;
};

/// @brief Type-erased core of `computeDem`. Lives in Dem.cpp so the templated
///        entry point in this header does not need the heavyweight simulator /
///        Stim includes.
DemData runComputeDem(const std::string &kernelName,
                      cudaq::quantum_platform &platform,
                      const cudaq::noise_model *noise,
                      const std::function<void()> &wrappedKernel);

} // namespace detail

/// @brief Run the DEM analysis engine over a CUDA-Q kernel.
///
/// The kernel is executed through the Stim simulator (forced via
/// `ScopedAnalysisSimulator`) under an `ExecutionContext` named "dem". The
/// simulator records the circuit, including `DETECTOR` and
/// `OBSERVABLE_INCLUDE` instructions emitted by the `qec.*` annotation ops,
/// and the policy-typed `finalize_simulation_circuit_impl` overload converts
/// it to a Stim DEM.
///
/// @note The active target / platform is not modified. Stim is used as an
///       internal analysis engine, never as a user-facing target.
///
/// @param kernel  A CUDA-Q kernel (any callable invocable with @p args).
/// @param noise   Optional noise model to attach for the analysis run. If
///                null, the kernel's `apply_noise` ops (if any) drive the DEM;
///                otherwise this noise model is layered on per
///                `cudaq::noise_model` semantics.
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
DemData computeDem(QuantumKernel &&kernel, const cudaq::noise_model *noise,
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
DemData computeDem(QuantumKernel &&kernel, Args &&...args) {
  return computeDem(std::forward<QuantumKernel>(kernel),
                    /*noise=*/nullptr, std::forward<Args>(args)...);
}

} // namespace cudaq_internal::analysis
