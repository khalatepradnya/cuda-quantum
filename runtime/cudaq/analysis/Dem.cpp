/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/analysis/dem.h"

#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"

#include "stim.h"

#include <stdexcept>
#include <utility>

namespace cudaq::analysis {

namespace dem {

nvqir::AnalysisScope make_scope(std::string plugin_name) {
  // `from_plugin` resolves the simulator via dlsym, then forwards to the
  // base `nvqir::AnalysisScope` constructor which claims the thread-local
  // slot. The `on_enter` hook drops any residue from a prior run on this
  // thread's copy of the plugin simulator; there is no `on_exit` because
  // the simulator is shared across CUDA-Q infrastructure and clearing
  // state on exit could break a follow-up `cudaq::sample` call.
  std::string name = "dem";
  return nvqir::AnalysisScope::from_plugin(
      std::move(name), std::move(plugin_name),
      {.on_enter = [](nvqir::CircuitSimulator &sim) {
        sim.resetCircuitRepr();
      }});
}

} // namespace dem

namespace detail {

stim::DetectorErrorModel runComputeDem(const std::string &kernelName,
                                       cudaq::quantum_platform &platform,
                                       const cudaq::noise_model *noise,
                                       const std::function<void()> &kernel,
                                       std::string plugin_name) {
  // RAII: scope releases the thread-local override (and runs the recorded-
  // circuit reset hook) on every exit path, including exceptions thrown
  // from the kernel.
  auto demScope = dem::make_scope(std::move(plugin_name));

  // `explicitMeasurements` keeps `mz` from deferring `M` ops to flush time.
  // The QEC dialect's `qec.detector(handle)` lowers to a `DETECTOR rec[-N]`
  // instruction that must reference an already-laid `M`; without explicit
  // measurements the detector would appear before its referent in the
  // recorded circuit, and `circuit_to_detector_error_model` would reject it.
  cudaq::ExecutionContext ctx("dem");
  ctx.explicitMeasurements = true;
  ctx.kernelName = kernelName;
  ctx.asyncExec = false;
  if (noise)
    ctx.noiseModel = noise;

  platform.with_execution_context(ctx, kernel);

  nvqir::CircuitSimulator *sim = demScope.simulator().getRecordedCircuit()
                                     ? &demScope.simulator()
                                     : nullptr;
  if (!sim)
    throw std::runtime_error(
        "`cudaq::analysis::compute_dem`: analysis simulator '" +
        std::string(demScope.name()) +
        "' did not produce a structured recorded circuit. DEM analysis "
        "requires a Stim-format recorded circuit; only the Stim NVQIR "
        "backend implements `getRecordedCircuit()` today.");

  // Defaults match Stim's CLI defaults except for `decompose_errors`, which
  // we leave off so the raw DEM is emitted. CUDA-QX may re-run the analyzer
  // with `decompose_errors=true` when feeding a matching decoder.
  return stim::ErrorAnalyzer::circuit_to_detector_error_model(
      *sim->getRecordedCircuit(),
      /*decompose_errors=*/false,
      /*fold_loops=*/false,
      /*allow_gauge_detectors=*/false,
      /*approximate_disjoint_errors_threshold=*/0,
      /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
}

} // namespace detail
} // namespace cudaq::analysis
