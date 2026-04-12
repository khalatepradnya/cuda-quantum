/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target stim %s -o %t && %t | FileCheck %s

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
      cudaq::sample({.shots = 10, .explicit_measurements = true},
                    rep_code_kernel);

  auto circuit = cudaq::getCircuitRepr();
  printf("%s", circuit.c_str());

  bool has_detector = circuit.find("DETECTOR") != std::string::npos;
  bool has_observable = circuit.find("OBSERVABLE_INCLUDE") != std::string::npos;

  if (has_detector && has_observable)
    printf("PASS\n");
  else
    printf("FAIL\n");

  return (has_detector && has_observable) ? 0 : 1;
}

// CHECK: DETECTOR rec[-3] rec[-2]
// CHECK: DETECTOR rec[-2] rec[-1]
// CHECK: OBSERVABLE_INCLUDE(0) rec[-3]
// CHECK: PASS
