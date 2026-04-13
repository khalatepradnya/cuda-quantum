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

#include <cudaq.h>
#include <cstdio>
#include <string>

extern "C" {
const char *__nvqir__getCircuitRepr();
}

// Distance-3 repetition code, 2 rounds of syndrome extraction.
// Round 1: measure stabilizer generators Z0Z1 and Z1Z2.
// Round 2: same stabilizers again.
// Cross-round detectors: parity of same stabilizer across rounds.
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
      cudaq::sample({.shots = 10, .explicit_measurements = true},
                    rep_code_2rounds);

  const char *repr = __nvqir__getCircuitRepr();
  std::string circuit = repr ? repr : "";
  printf("%s", circuit.c_str());

  // Count detectors and observables in the output
  int detCount = 0, obsCount = 0;
  std::string::size_type pos = 0;
  while ((pos = circuit.find("DETECTOR", pos)) != std::string::npos) {
    detCount++;
    pos += 8;
  }
  pos = 0;
  while ((pos = circuit.find("OBSERVABLE_INCLUDE", pos)) != std::string::npos) {
    obsCount++;
    pos += 18;
  }

  // 4 detectors (2 round-1, 2 cross-round) + 1 observable
  printf("detectors=%d observables=%d\n", detCount, obsCount);
  if (detCount == 4 && obsCount == 1)
    printf("PASS\n");
  else
    printf("FAIL\n");

  return (detCount == 4 && obsCount == 1) ? 0 : 1;
}

// CHECK: DETECTOR
// CHECK: DETECTOR
// CHECK: DETECTOR
// CHECK: DETECTOR
// CHECK: OBSERVABLE_INCLUDE
// CHECK: detectors=4 observables=1
// CHECK: PASS
