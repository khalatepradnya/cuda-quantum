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
#include "cudaq/analysis/StimDem.h"

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
// runRecordDem — Option C core: drives the kernel through the Stim analysis
//                simulator and returns the resulting DEM by value, with no
//                text serialisation between the recorder and the caller.
// ===========================================================================

stim::DetectorErrorModel
runRecordDem(const std::string &kernelName,
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
  platform.with_execution_context(ctx, wrappedKernel);

  // Read the structured `stim::Circuit` directly off the (still-active)
  // analysis simulator -- no text serialisation, no re-parse. The
  // ScopedAnalysisSimulator guard keeps Stim as the active simulator until
  // this function returns; the recordedCircuit storage is owned by Stim and
  // valid for the lifetime of the active simulator instance, which is
  // longer than this function call.
  nvqir::CircuitSimulator *sim = nvqir::getCircuitSimulatorInternal();
  const stim::Circuit *recorded = sim->getRecordedCircuit();
  if (!recorded)
    throw std::runtime_error(
        "cudaq::analysis::record_dem: simulator '" + sim->name() +
        "' did not provide a structured recorded circuit. DEM analysis "
        "requires a Stim-format recorded circuit (only the Stim NVQIR "
        "backend supports this today).");

  // Run Stim's error analyzer directly on the structured circuit. Defaults
  // match Stim's CLI defaults except for `decompose_errors`, which we leave
  // off so the raw DEM is emitted; CUDA-QX may set it to true when consuming
  // the DEM for matching decoders.
  (void)kernelName;
  return stim::ErrorAnalyzer::circuit_to_detector_error_model(
      *recorded,
      /*decompose_errors=*/false,
      /*fold_loops=*/false,
      /*allow_gauge_detectors=*/false,
      /*approximate_disjoint_errors_threshold=*/0,
      /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
}

// ===========================================================================
// runComputeDem — Option A wire-protocol shim: calls runRecordDem and
//                 packages the DEM as text + counts in DemData. Kept for
//                 backwards compatibility with the existing Option A
//                 prototype while Option C is being prototyped end-to-end.
// ===========================================================================

DemData runComputeDem(const std::string &kernelName,
                      cudaq::quantum_platform &platform,
                      const cudaq::noise_model *noise,
                      const std::function<void()> &wrappedKernel) {
  stim::DetectorErrorModel dem =
      runRecordDem(kernelName, platform, noise, wrappedKernel);

  DemData result;
  std::stringstream demStream;
  demStream << dem;
  result.demText = demStream.str();
  result.numDetectors = dem.count_detectors();
  result.numObservables = dem.count_observables();
  return result;
}

} // namespace detail
} // namespace cudaq::analysis
