/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file Dem.cpp
/// @brief Out-of-line implementation of the DEM analysis engine.
///
/// Lives in libcudaq-analysis so that:
///   1. The simulator (which sees `dem_policy` via ADL through the CPO in
///      `policy_dispatch.h`) does not need to depend on Stim headers.
///   2. The CUDA-QX wrapper (`cudaq::qec::dem_from_kernel`) can include
///      `cudaq_internal/analysis/Dem.h` (light) without dragging in Stim.

#include "cudaq_internal/analysis/Dem.h"

#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"

#include "stim.h"

#include <sstream>
#include <stdexcept>

// =============================================================================
// NVQIR forward declarations
// =============================================================================
//
// These are the analysis-engine-side counterparts of
// `nvqir::switchToResourceCounterSimulator` / `stopUsingResourceCounterSimulator`.
// They live in `runtime/nvqir/NVQIR.cpp`. Forward-declared here so libcudaq-analysis
// does not pull a public NVQIR analysis header just to wire two functions; if
// a third analysis engine appears, this should be promoted to a small shared
// header (e.g., `runtime/nvqir/AnalysisSim.h`).
namespace nvqir {
void pushAnalysisSimulator(const std::string &pluginName);
void popAnalysisSimulator();
} // namespace nvqir

namespace cudaq_internal::analysis {

// =============================================================================
// Thread-local result slot
// =============================================================================

namespace detail {
namespace {
thread_local DemData *g_resultSlot = nullptr;
} // namespace

DemData *getResultSlot() { return g_resultSlot; }
void setResultSlot(DemData *slot) { g_resultSlot = slot; }

// ===========================================================================
// ScopedAnalysisSimulator — RAII override of the active NVQIR simulator
// ===========================================================================

ScopedAnalysisSimulator::ScopedAnalysisSimulator() {
  // v1 hard-codes "stim" as the only analysis backend. See FIXME on the
  // class declaration in Dem.h; the policy CPO direction is to derive this
  // from the policy struct itself rather than baking the choice in here.
  nvqir::pushAnalysisSimulator("stim");
}

ScopedAnalysisSimulator::~ScopedAnalysisSimulator() {
  // Mirror the resource-counter pattern: best-effort restore. NVQIR's
  // `popAnalysisSimulator` is no-op-safe if push failed.
  nvqir::popAnalysisSimulator();
}

// ===========================================================================
// runComputeDem — type-erased core of the templated computeDem entry point
// ===========================================================================

DemData runComputeDem(const std::string &kernelName,
                      cudaq::quantum_platform &platform,
                      const cudaq::noise_model *noise,
                      const std::function<void()> &wrappedKernel) {
  // FIXME(runtime-team): the steps below are the boilerplate every analysis
  // engine will repeat (sim override + ExecutionContext + result slot). The
  // policy-based dispatch direction would push this into a CPO such as
  // `cudaq::policies::run_analysis(policy, kernel)`, leaving each engine to
  // declare only its policy struct + finalize overload.
  ScopedAnalysisSimulator simGuard;

  cudaq::ExecutionContext ctx("dem");
  // Stim's chronological measurement record is only stable when the runtime
  // commits an `M` op for every `mz` (no batching across shots). The Stim
  // simulator's `mz` -> `M` lowering is gated on this flag, and the
  // `qec.detector` / `qec.logical_observable` lowerings emit
  // `rec[-N]`-style references that depend on that ordering. See
  // `StimCircuitSimulator::flushPendingSampleMeasurements` and the comments
  // around `getMeasureIndex` for the full story.
  ctx.explicitMeasurements = true;

  if (noise)
    // ExecutionContext::noiseModel is non-const; the analysis engine treats
    // it as logically read-only. const_cast here is safe and localized.
    ctx.noiseModel = const_cast<cudaq::noise_model *>(noise);

  DemData result;
  setResultSlot(&result);
  try {
    platform.with_execution_context(ctx, wrappedKernel);
  } catch (...) {
    setResultSlot(nullptr);
    throw;
  }
  setResultSlot(nullptr);

  // Surface the kernel name for diagnostics; the result struct does not
  // carry it today but downstream callers (CUDA-QX) may want to attach it
  // to error messages.
  (void)kernelName;
  return result;
}

} // namespace detail

// =============================================================================
// finalize_simulation_circuit_impl — the dem_policy CPO overload
// =============================================================================
//
// This is the hidden friend declared on `dem_policy` in DemPolicy.h. It is
// reached via ADL from the `finalize_simulation_circuit` CPO in
// `cudaq/algorithms/policy_cpos.h`, which is invoked by
// `nvqir::CircuitSimulator::finalizeExecutionContext` when the active context
// has `name == "dem"` (registered in `policy_dispatch.h`).
//
// Hidden-friend definitions live at namespace scope but are only callable via
// ADL through `dem_policy` arguments — that is exactly the desired behavior
// here: the simulator's CPO call discovers this overload through `policy`'s
// type, and no other call path can reach it.

void finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                      const dem_policy &policy,
                                      cudaq::ExecutionContext &ctx) {
  (void)policy;
  (void)ctx;

  DemData *slot = detail::getResultSlot();
  if (!slot)
    throw std::runtime_error(
        "cudaq_internal::analysis::dem_policy: result slot not set. The "
        "dem CPO must be invoked through "
        "cudaq_internal::analysis::computeDem(...) so the result destination "
        "is in scope.");

  std::string repr = sim.getCircuitRepr();
  if (repr.empty())
    throw std::runtime_error(
        "cudaq_internal::analysis::dem_policy: simulator '" + sim.name() +
        "' produced an empty circuit repr. DEM analysis requires a "
        "Stim-format recorded circuit (only the Stim NVQIR backend supports "
        "this today).");

  // Parse the recorded Stim circuit and run Stim's error analyzer to obtain
  // the DEM. The defaults below match Stim's CLI defaults except for
  // `decompose_errors`, which we leave off so the v1 demo emits the raw DEM;
  // CUDA-QX may set it to true when consuming the DEM for matching decoders.
  stim::Circuit circuit(repr.c_str());
  stim::DetectorErrorModel dem =
      stim::ErrorAnalyzer::circuit_to_detector_error_model(
          circuit,
          /*decompose_errors=*/false,
          /*fold_loops=*/false,
          /*allow_gauge_detectors=*/false,
          /*approximate_disjoint_errors_threshold=*/0,
          /*ignore_decomposition_failures=*/false,
          /*block_decomposition_from_introducing_remnant_edges=*/false);

  std::stringstream demStream;
  demStream << dem;
  slot->demText = demStream.str();
  slot->numDetectors = circuit.count_detectors();
  slot->numObservables = circuit.count_observables();
}

} // namespace cudaq_internal::analysis
