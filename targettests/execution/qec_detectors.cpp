/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target stim %s -o %t && %t | FileCheck %s

// End-to-end test: detector and logical_observable declarations compile,
// lower through QEC -> QIR, and execute on the Stim backend without error.
// Noise-free repetition code: all measurements are 0, all shots identical.

#include <cudaq.h>
#include <cstdio>

auto rep_code_kernel = []() __qpu__ {
  cudaq::qvector q(3);

  cudaq::cx(q[0], q[1]);
  cudaq::cx(q[1], q[2]);

  auto s0 = cudaq::mz(q[0]);
  auto s1 = cudaq::mz(q[1]);
  auto s2 = cudaq::mz(q[2]);

  cudaq::detector(s0, s1);
  cudaq::detector(s1, s2);
  cudaq::logical_observable(s0);
};

int main() {
  auto result =
      cudaq::sample({.shots = 100, .explicit_measurements = true},
                    rep_code_kernel);

  // Noise-free: all qubits start |0>, CNOT copies |0>, all measurements = 0.
  // Every shot should produce the same bitstring "000".
  auto mostProbable = result.most_probable();
  printf("most_probable: %s\n", mostProbable.c_str());

  bool allZero = (mostProbable == "000");
  bool singleOutcome = (result.size() == 1);

  if (allZero && singleOutcome)
    printf("PASS\n");
  else
    printf("FAIL: outcomes=%zu\n", result.size());

  return (allZero && singleOutcome) ? 0 : 1;
}

// CHECK: most_probable: 000
// CHECK: PASS
