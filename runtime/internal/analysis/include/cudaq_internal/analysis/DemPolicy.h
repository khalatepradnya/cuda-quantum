/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file DemPolicy.h
/// @brief Light header defining the `dem_policy` tag and `DemData` result
///        struct. Split out of `Dem.h` so that `cudaq/algorithms/policy_dispatch.h`
///        (which is transitively included by every NVQIR simulator) can pull
///        in the policy without dragging in `cudaq/platform.h` and the kernel
///        builder.
///
/// The full engine entry point (`computeDem`, `ScopedAnalysisSimulator`) lives
/// in `Dem.h`, which includes this header.

#pragma once

#include <cstddef>
#include <string>

namespace cudaq {
class ExecutionContext;
} // namespace cudaq

namespace nvqir {
class CircuitSimulator;
} // namespace nvqir

namespace cudaq_internal::analysis {

// =============================================================================
// DemData — analysis engine result type
// =============================================================================

/// @brief DEM (Detector Error Model) data produced by the analysis engine.
///
/// For v1 this carries the DEM in Stim's textual representation. CUDA-QX's
/// `cudaq::qec::dem_from_kernel(...)` parses it into the dense
/// `cudaq::qec::detector_error_model` struct (H, error_rates,
/// observables_flips_matrix). A future revision will likely return the dense
/// form directly to avoid the round-trip through text.
struct DemData {
  /// @brief DEM as emitted by Stim's `circuit_to_detector_error_model`,
  ///        printed via the standard Stim DEM text format.
  std::string demText;

  /// @brief Number of declared detectors in the analysis circuit.
  std::size_t numDetectors = 0;

  /// @brief Number of declared logical observables.
  std::size_t numObservables = 0;
};

// =============================================================================
// dem_policy — tag + options for DEM analysis
// =============================================================================

namespace detail {

/// @brief Thread-local destination for the DemData produced by the simulator's
/// `finalize_simulation_circuit_impl(sim, dem_policy, ctx)` overload.
///
/// FIXME(runtime-team): The CPO dispatch in `policy_dispatch.h::withPolicy`
/// constructs the policy via its default ctor inside the registry callback,
/// so callers cannot attach a result destination to the policy instance the
/// simulator sees. Until `withPolicy` accepts a caller-provided policy
/// instance (or an equivalent mechanism such as
/// `cudaq::policies::current_policy<P>()` lands), DEM uses this thread-local
/// to ferry the result back to `computeDem`. Replace with policy-carried
/// state once the policy infra grows.
DemData *getResultSlot();
void setResultSlot(DemData *slot);

} // namespace detail

/// @brief Tag + options for DEM analysis. Modeled on `cudaq::sample_policy`.
///
/// The hidden-friend `finalize_simulation_circuit_impl` overload is found via
/// ADL through this struct's namespace (`cudaq_internal::analysis`). It is
/// invoked by `nvqir::CircuitSimulator::finalizeExecutionContext` when the
/// active execution context has `name == "dem"`.
struct dem_policy {
  /// @brief Conceptual result type for this policy. Semantically, dem_policy
  ///        produces `DemData`. In the v1 implementation the friend overload
  ///        returns `void` and writes to the thread-local result slot (see
  ///        `detail::getResultSlot`) because the existing
  ///        `finalize_simulation_circuit` visitor in
  ///        `nvqir::CircuitSimulator::finalizeExecutionContext` only declares
  ///        handlers for `sample_result` and `void_result`. Adding a third
  ///        handler would couple every simulator's finalize visitor to the
  ///        analysis module, which is exactly the kind of bloat the policy
  ///        direction aims to avoid. See FIXME on `detail::getResultSlot`.
  using result_type = DemData;

  // Hidden friend: discovered via ADL through cudaq_internal::analysis.
  // Returns void in v1; the result is written to detail::getResultSlot() and
  // the caller (`runComputeDem`) reads it back. When the policy CPO grows a
  // typed result channel that does not require per-type visitor handlers,
  // change this signature to `DemData` and remove the TLS slot.
  friend void
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const dem_policy &policy,
                                   cudaq::ExecutionContext &ctx);
};

} // namespace cudaq_internal::analysis
