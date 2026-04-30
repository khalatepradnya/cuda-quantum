/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file DemPolicy.h
/// @brief Light header defining the `dem_policy` tag and `DemData` result
///        struct. Split out of `Dem.h` so that
///        `cudaq/algorithms/policy_dispatch.h` (which is transitively included
///        by every NVQIR simulator) can pull in the policy without dragging in
///        `cudaq/platform.h` and the kernel builder.
///
/// The full engine entry point (`computeDem`, `ScopedAnalysisSimulator`) lives
/// in `Dem.h`, which includes this header.

#pragma once

#include <cstddef>
#include <string>

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
// dem_policy — tag for DEM analysis
// =============================================================================

/// @brief Tag struct for DEM analysis, registered in `policy_dispatch.h`.
///
/// Unlike `sample_policy` (which carries a hidden-friend
/// `finalize_simulation_circuit_impl` overload that the CPO invokes during
/// simulator finalization), `dem_policy` deliberately has NO custom finalize
/// overload. The CPO's `has_sim_custom_finalize<dem_policy>` evaluates to
/// `false`, so the simulator's `finalizeExecutionContext` falls through to the
/// default no-op path for "other_policies".
///
/// DEM extraction happens *after* `with_execution_context` returns, in
/// `runComputeDem` (Dem.cpp). The Stim simulator's `recordedCircuit` survives
/// finalization (it is intentionally not cleared in `deallocateStateImpl`), so
/// `getCircuitRepr()` remains valid at that point.
///
/// This design avoids the link-time coupling that a hidden-friend overload
/// would create: every NVQIR simulator would need to link `libcudaq-analysis`
/// just because the CPO concept check sees the friend declaration in this
/// header (which is transitively included by every simulator via
/// `policy_dispatch.h` → `CircuitSimulator.h`).
///
/// FIXME(runtime-team): when the policy CPO grows a mechanism for
/// policy-specific finalization that does not require every simulator to
/// resolve the symbol at link time (e.g., a virtual dispatch or a plugin
/// registration point), revisit whether DEM extraction should move back into
/// the finalize path.
struct dem_policy {};

} // namespace cudaq_internal::analysis
