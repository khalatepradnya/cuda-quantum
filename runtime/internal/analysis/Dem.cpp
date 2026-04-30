/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file Dem.cpp
/// @brief Out-of-line implementation of the DEM analysis engine.
///
/// Lives in libcudaq-analysis so that:
///   1. The simulator (which sees `dem_policy` via `policy_dispatch.h`)
///      does not need to depend on Stim headers.
///   2. The CUDA-QX wrapper (`cudaq::qec::dem_from_kernel`) can include
///      `cudaq/analysis/Dem.h` (light) without dragging in Stim.

#include "cudaq/analysis/Dem.h"

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
// `nvqir::switchToResourceCounterSimulator` /
// `stopUsingResourceCounterSimulator`. They live in `runtime/nvqir/NVQIR.cpp`.
// Forward-declared here so libcudaq-analysis does not pull a public NVQIR
// analysis header just to wire a few functions; if a third analysis engine
// appears, promote these to a small shared header (e.g.,
// `runtime/nvqir/AnalysisSim.h`).
namespace nvqir {
void pushAnalysisSimulator(const std::string &pluginName);
void popAnalysisSimulator();
CircuitSimulator *getCircuitSimulatorInternal();
} // namespace nvqir

namespace cudaq::analysis {

// ===========================================================================
// ScopedAnalysisSimulator — RAII override of the active NVQIR simulator
// ===========================================================================

namespace detail {

ScopedAnalysisSimulator::ScopedAnalysisSimulator() {
  // v1 hard-codes "stim" as the only analysis backend. See FIXME on the
  // class declaration in Dem.h; the policy CPO direction is to derive this
  // from the policy struct itself rather than baking the choice in here.
  nvqir::pushAnalysisSimulator("stim");
}

ScopedAnalysisSimulator::~ScopedAnalysisSimulator() {
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
  // engine will repeat (sim override + ExecutionContext + post-exec
  // extraction). The policy-based dispatch direction would push this into a
  // CPO such as `cudaq::policies::run_analysis(policy, kernel)`, leaving
  // each engine to declare only its policy struct + extraction logic.
  ScopedAnalysisSimulator simGuard;

  cudaq::ExecutionContext ctx("dem");
  ctx.explicitMeasurements = true;

  if (noise)
    ctx.noiseModel = const_cast<cudaq::noise_model *>(noise);

  // Run the kernel. The simulator records the full circuit including
  // DETECTOR / OBSERVABLE_INCLUDE instructions emitted by the qec.* ops.
  // The "dem" entry in the withPolicy registry dispatches to dem_policy{},
  // which has no custom finalize overload — the CPO falls through to the
  // default no-op. This is intentional: DEM extraction happens below,
  // after the kernel finishes, because the Stim simulator's
  // recordedCircuit survives finalization.
  platform.with_execution_context(ctx, wrappedKernel);

  // Extract the recorded Stim circuit from the (still-active) analysis
  // simulator. The ScopedAnalysisSimulator guard keeps Stim as the active
  // simulator until this function returns.
  nvqir::CircuitSimulator *sim = nvqir::getCircuitSimulatorInternal();
  std::string repr = sim->getCircuitRepr();
  if (repr.empty())
    throw std::runtime_error(
        "cudaq::analysis::computeDem: simulator '" + sim->name() +
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

  DemData result;
  std::stringstream demStream;
  demStream << dem;
  result.demText = demStream.str();
  result.numDetectors = circuit.count_detectors();
  result.numObservables = circuit.count_observables();

  (void)kernelName;
  return result;
}

} // namespace detail
} // namespace cudaq::analysis
