/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <map>
#include <vector>

namespace cudaq {

/// Backend-owned QEC data produced by the Stim simulator.
/// Retrievable via platform.query<StimQECData>() after kernel execution.
struct StimQECData {
  /// Detector matrix rows. Row i = set of chronological measurement indices
  /// that participate in detector i.
  std::vector<std::vector<std::size_t>> detectorRows;

  /// Observable matrix rows. Key = observable index, value = set of
  /// chronological measurement indices for that observable.
  std::map<std::size_t, std::vector<std::size_t>> observableRows;

  /// Total number of measurements in the circuit.
  std::size_t totalMeasurements = 0;
};

} // namespace cudaq
