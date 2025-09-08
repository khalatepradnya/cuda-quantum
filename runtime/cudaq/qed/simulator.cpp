/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "error_detection.h"

/// Simply return false

__qpu__ bool cudaq::qed::md(cudaq::qubit &q) { return false; }

__qpu__ std::vector<bool> cudaq::qed::md(cudaq::qview<> q) {
  // return std::vector<bool>(q.size(), false);
  std::vector<bool> results(q.size());
  for (int i = 0; i < q.size(); i++) {
    results[i] = false;
  }
  return results;
}
