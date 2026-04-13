/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/measure_result.h"

/// \file
/// cudaq::detail construction helpers for measure_result and measure_vector.
/// This header is NOT part of the user API (cudaq::detail is explicitly
/// non-public per CppAPICodingStyle.md section 3.1.3). It exists so that
/// the runtime (NVQIR, measurement functions) can construct these types.

namespace cudaq::detail {

struct MeasureResultShim {
  static measure_result create(std::int64_t val, std::int64_t id) {
    return measure_result(val, id);
  }
  static measure_result create(std::int64_t val) { return measure_result(val); }
  static measure_vector make_vector(const measure_result *data,
                                    std::size_t size) {
    return measure_vector(data, size);
  }
};

} // namespace cudaq::detail
