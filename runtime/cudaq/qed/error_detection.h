/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qubit_qis.h"
#include <vector>

namespace cudaq::qed {

// Detect errors on input qubits, a value of `true` means error detected,
// `false` means no error.
__qpu__ std::vector<bool> md(cudaq::qview<> q);

// Overload for single qubit
__qpu__ bool md(cudaq::qubit &q);

} // namespace cudaq::qed
