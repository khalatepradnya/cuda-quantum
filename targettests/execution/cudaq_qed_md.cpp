/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN:nvq++ -lcudaq-qed-simulator %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/qed/error_detection.h>

__qpu__ int foo() {
  cudaq::qubit q;
  h(q);
  if (cudaq::qed::md(q)) {
    return -1;
  }
  return mz(q);
}

int main() {
  auto results = cudaq::run(10, foo);
  for (auto r : results) {
    printf("Result: %d\n", r);
  }
  return 0;
}

// CHECK-NOT: Result: -1
