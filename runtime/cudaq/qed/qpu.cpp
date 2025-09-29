/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "error_detection.h"
#include <cudaq.h>

// interface function that the device can implement
extern "C" void device_md(cudaq::qudit<2> *q, bool *result, std::uint64_t rsize);

/// Call to device
__qpu__ std::vector<bool> cudaq::qed::md(cudaq::qview<> q) {
  // create the boolean result vector
  std::vector<bool> result(q.size());
  // call the device function
  // NB: you cannot pass a vector<T>! (Yet)
  cudaq::device_call(device_md, q, result.data(), result.size());
  return result;
}
