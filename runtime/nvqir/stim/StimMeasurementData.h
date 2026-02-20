/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <unordered_map>

namespace cudaq {

/// @brief Per-measurement metadata produced by the Stim backend.
/// Keyed by uniqueId (from `measure_result::getUniqueId()`).
struct StimMeasurementData {
  struct MeasurementInfo {
    std::size_t qubit_index;
    // can add more fields here...
  };

  std::unordered_map<int, MeasurementInfo> measurements;

  /// @brief Lookup measurement info by uniqueId.
  const MeasurementInfo *lookup(int uniqueId) const {
    auto it = measurements.find(uniqueId);
    if (it == measurements.end())
      return nullptr;
    return &it->second;
  }
};

} // namespace cudaq
