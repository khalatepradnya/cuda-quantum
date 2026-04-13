/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>

namespace cudaq {

/// Base interface for QEC metadata produced by backends after kernel execution.
/// Backends that support detector and logical observable declarations should
/// provide a concrete subclass. Retrieved via platform.query<ConcreteType>().
///
/// Note: `std::any_cast` does not support polymorphic down-casting, so
/// query<qec_metadata>() cannot be used to retrieve a derived type stored
/// as itself. Users must query with the concrete backend type (e.g.,
/// query<StimQECData>()). This base class serves as a documented contract
/// for what QEC backends must expose.
struct qec_metadata {
  virtual ~qec_metadata() = default;

  /// Number of detectors declared in the kernel.
  virtual std::size_t num_detectors() const = 0;

  /// Number of distinct logical observables declared in the kernel.
  virtual std::size_t num_observables() const = 0;

  /// Total number of measurements in the circuit.
  virtual std::size_t num_measurements() const = 0;
};

} // namespace cudaq
