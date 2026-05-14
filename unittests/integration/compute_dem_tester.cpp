/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/analysis/dem.h"
#include "nvqir/AnalysisScope.h"
#include "nvqir/CircuitSimulator.h"

#include <stdexcept>

// The DEM analysis path is only implemented on the Stim NVQIR backend (the
// only one that produces a non-null `getRecordedCircuit()`). Compile out
// everywhere else so the rest of the test runtime stays buildable.
#ifdef CUDAQ_BACKEND_STIM

CUDAQ_TEST(ComputeDemTester, trivialKernelEmptyDem) {
  auto trivialKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };

  auto dem = cudaq::analysis::compute_dem(trivialKernel);
  EXPECT_EQ(dem.count_detectors(), 0u);
  EXPECT_EQ(dem.count_observables(), 0u);
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

// TODO(followups.md "compute_dem MLIR-mode test"): a richer positive-path
// test that exercises `cudaq::detector(...)` lowering through the QEC dialect
// needs an MLIR-mode integration test. The integration suite here is built
// with `-DCUDAQ_LIBRARY_MODE`, which gates `cudaq::detector` / `cudaq::
// logical_observable` out of `qubit_qis.h`. Library-mode kernels run through
// the ExecutionManager interpreter and never hit the MLIR JIT pipeline that
// lowers `qec.*` ops, so even if the symbol resolved at compile time the op
// would never reach Stim's recorded circuit. Cover the with-detector case
// via a lit test or an MLIR-mode integration test once one exists for QEC.

CUDAQ_TEST(ComputeDemTester, releasesScopeOnException) {
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());

  auto throwingKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    throw std::runtime_error("kernel boom");
  };

  EXPECT_THROW(cudaq::analysis::compute_dem(throwingKernel),
               std::runtime_error);

  // RAII on the analysis scope should have released the slot even though
  // the body threw.
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());

  // A subsequent compute_dem on the same thread must work.
  auto trivialKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };
  auto dem = cudaq::analysis::compute_dem(trivialKernel);
  EXPECT_EQ(dem.count_detectors(), 0u);
}

CUDAQ_TEST(ComputeDemTester, demScopeIsActive) {
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
  {
    auto s = cudaq::analysis::dem::make_scope();
    EXPECT_TRUE(nvqir::AnalysisScope::is_active());
    EXPECT_EQ(s.name(), "dem");
    EXPECT_EQ(s.simulator().name(), "stim");
  }
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(ComputeDemTester, demScopeNestedThrows) {
  auto outer = cudaq::analysis::dem::make_scope();
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());

  EXPECT_THROW(cudaq::analysis::dem::make_scope(), std::runtime_error);

  // Outer scope is still the active one after the failed nest attempt.
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(ComputeDemTester, makeScopeRejectsUnknownPlugin) {
  // `dem::make_scope` exposes the plugin name as the documented extension
  // point. Resolving an unknown plugin must surface as a runtime error
  // rather than a null-deref later in `compute_dem`.
  EXPECT_THROW(cudaq::analysis::dem::make_scope("nonexistent_plugin"),
               std::runtime_error);
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

#endif // CUDAQ_BACKEND_STIM
