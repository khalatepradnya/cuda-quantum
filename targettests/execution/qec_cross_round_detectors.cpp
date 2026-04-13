/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target stim %s -o %t && %t | FileCheck %s

// Cross-round detectors are the core QEC pattern: comparing the same
// stabilizer measurement across consecutive syndrome extraction rounds.
// This test validates that compiled-mode lowering correctly resolves
// measurement indices for detectors spanning multiple rounds, and that
// logical_observable with an explicit index works end-to-end.
//
// Noise-free distance-3 repetition code, 2 rounds of syndrome extraction.
// All measurements should be 0 on every shot.

#include <cudaq.h>
#include <cstdio>

auto rep_code_2rounds = []() __qpu__ {
  cudaq::qvector data(3);
  cudaq::qubit anc0, anc1;

  // --- Round 1 ---
  cudaq::cx(data[0], anc0);
  cudaq::cx(data[1], anc0);
  auto s0_r1 = cudaq::mz(anc0);
  cudaq::reset(anc0);

  cudaq::cx(data[1], anc1);
  cudaq::cx(data[2], anc1);
  auto s1_r1 = cudaq::mz(anc1);
  cudaq::reset(anc1);

  // Round 1 detectors: on freshly initialized |000>, stabilizers are
  // deterministic (both = 0), so single-measurement detectors are valid.
  cudaq::detector(s0_r1);
  cudaq::detector(s1_r1);

  // --- Round 2 ---
  cudaq::cx(data[0], anc0);
  cudaq::cx(data[1], anc0);
  auto s0_r2 = cudaq::mz(anc0);
  cudaq::reset(anc0);

  cudaq::cx(data[1], anc1);
  cudaq::cx(data[2], anc1);
  auto s1_r2 = cudaq::mz(anc1);
  cudaq::reset(anc1);

  // Cross-round detectors: parity of same stabilizer across rounds.
  cudaq::detector(s0_r1, s0_r2);
  cudaq::detector(s1_r1, s1_r2);

  // --- Data qubit readout ---
  auto d0 = cudaq::mz(data[0]);
  auto d1 = cudaq::mz(data[1]);
  auto d2 = cudaq::mz(data[2]);

  // Logical observable: XOR of all data qubits = logical Z.
  cudaq::logical_observable(d0, d1, d2);
};

int main() {
  auto result =
      cudaq::sample({.shots = 100, .explicit_measurements = true},
                    rep_code_2rounds);

  // 7 measurements total: s0_r1, s1_r1, s0_r2, s1_r2, d0, d1, d2.
  // Noise-free |000>: all measurements = 0, every shot identical.
  auto mostProbable = result.most_probable();
  printf("most_probable: %s\n", mostProbable.c_str());

  bool allZero = (mostProbable == "0000000");
  bool singleOutcome = (result.size() == 1);

  if (allZero && singleOutcome)
    printf("PASS\n");
  else
    printf("FAIL: most_probable=%s outcomes=%zu\n", mostProbable.c_str(),
           result.size());

  return (allZero && singleOutcome) ? 0 : 1;
}

// CHECK: most_probable: 0000000
// CHECK: PASS
