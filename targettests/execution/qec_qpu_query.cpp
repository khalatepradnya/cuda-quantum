/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target stim %s -o %t && %t | FileCheck %s

// Verify that QEC data is accessible via the QPU metadata query API
// (platform.query<StimQECData>()) after kernel execution, without
// relying on ExecutionContext fields.

#include <cudaq.h>
#include <cstdio>
#include <nvqir/stim/StimQECData.h>

// Distance-3 repetition code, 2 rounds.
// 4 detectors (2 round-1 + 2 cross-round) + 1 observable.
// 7 measurements total (2 ancilla r1, 2 ancilla r2, 3 data).
auto rep_code_2rounds = []() __qpu__ {
  cudaq::qvector data(3);
  cudaq::qubit anc0, anc1;

  // Round 1
  cudaq::cx(data[0], anc0);
  cudaq::cx(data[1], anc0);
  auto s0_r1 = cudaq::mz(anc0);
  cudaq::reset(anc0);

  cudaq::cx(data[1], anc1);
  cudaq::cx(data[2], anc1);
  auto s1_r1 = cudaq::mz(anc1);
  cudaq::reset(anc1);

  cudaq::detector(s0_r1);
  cudaq::detector(s1_r1);

  // Round 2
  cudaq::cx(data[0], anc0);
  cudaq::cx(data[1], anc0);
  auto s0_r2 = cudaq::mz(anc0);
  cudaq::reset(anc0);

  cudaq::cx(data[1], anc1);
  cudaq::cx(data[2], anc1);
  auto s1_r2 = cudaq::mz(anc1);
  cudaq::reset(anc1);

  cudaq::detector(s0_r1, s0_r2);
  cudaq::detector(s1_r1, s1_r2);

  auto d0 = cudaq::mz(data[0]);
  auto d1 = cudaq::mz(data[1]);
  auto d2 = cudaq::mz(data[2]);

  cudaq::logical_observable(d0, d1, d2);
};

int main() {
  cudaq::sample({.shots = 10, .explicit_measurements = true},
                rep_code_2rounds);

  auto &platform = cudaq::get_platform();
  auto *qecData = platform.query<cudaq::StimQECData>();

  if (!qecData) {
    printf("FAIL: query<StimQECData> returned null\n");
    return 1;
  }

  printf("detectors=%zu observables=%zu measurements=%zu\n",
         qecData->detectorRows.size(), qecData->observableRows.size(),
         qecData->totalMeasurements);

  bool pass = true;
  if (qecData->detectorRows.size() != 4) {
    printf("FAIL: expected 4 detectors, got %zu\n",
           qecData->detectorRows.size());
    pass = false;
  }
  if (qecData->observableRows.size() != 1) {
    printf("FAIL: expected 1 observable, got %zu\n",
           qecData->observableRows.size());
    pass = false;
  }
  if (qecData->totalMeasurements != 7) {
    printf("FAIL: expected 7 measurements, got %zu\n",
           qecData->totalMeasurements);
    pass = false;
  }

  // Observable 0 should reference 3 data qubit measurements (indices 4,5,6).
  auto it = qecData->observableRows.find(0);
  if (it == qecData->observableRows.end() || it->second.size() != 3) {
    printf("FAIL: observable 0 should have 3 measurement indices\n");
    pass = false;
  }

  if (pass)
    printf("PASS\n");

  return pass ? 0 : 1;
}

// CHECK: detectors=4 observables=1 measurements=7
// CHECK: PASS
