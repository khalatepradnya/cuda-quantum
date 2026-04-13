/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>

namespace cudaq {

extern "C" {
bool __nvqpp__MeasureResultBoolConversion(int);
}

namespace detail {
struct MeasureResultShim;
} // namespace detail

/// The return type of a qubit measurement. Represents a post-measurement
/// classical value tagged with a unique identity for backend correlation.
/// Copy and move are allowed (this is classical data, not a quantum resource).
/// Construction is restricted to the runtime via cudaq::detail.
class measure_result {
  std::int64_t value = 0;

  /// Lookup key for backend-specific metadata. INT64_MAX means unassigned.
  std::int64_t unique_id = std::numeric_limits<std::int64_t>::max();

  explicit measure_result(std::int64_t val) : value(val) {}
  explicit measure_result(std::int64_t val, std::int64_t id)
      : value(val), unique_id(id) {}

  friend struct detail::MeasureResultShim;

public:
  measure_result() = delete;
  measure_result(const measure_result &) = default;
  measure_result(measure_result &&) = default;
  measure_result &operator=(const measure_result &) = default;
  measure_result &operator=(measure_result &&) = default;

  std::int64_t get_unique_id() const { return unique_id; }

#ifdef CUDAQ_LIBRARY_MODE
  operator bool() const { return __nvqpp__MeasureResultBoolConversion(value); }
#else
  operator bool() const { return value == 1; }
#endif
  explicit operator int() const { return static_cast<int>(value); }
  explicit operator double() const { return static_cast<double>(value); }

  friend bool operator==(const measure_result &m1, const measure_result &m2) {
    return m1.value == m2.value;
  }
  friend bool operator==(const measure_result &m, bool b) {
    return static_cast<bool>(m) == b;
  }
  friend bool operator==(bool b, const measure_result &m) {
    return b == static_cast<bool>(m);
  }

  friend bool operator!=(const measure_result &m1, const measure_result &m2) {
    return m1.value != m2.value;
  }
  friend bool operator!=(const measure_result &m, bool b) {
    return static_cast<bool>(m) != b;
  }
  friend bool operator!=(bool b, const measure_result &m) {
    return b != static_cast<bool>(m);
  }
};

/// Immutable collection of measurement results. Maps to `!quake.measurements`
/// in the IR, analogous to how `qvector` maps to `!quake.veq`. Replaces
/// `std::vector<measure_result>` which exposed meaningless operations
/// (push_back, resize, insert) that cannot be lowered to IR.
///
/// No default construction, no copy, no move, no assignment - like `qvector`.
/// Created only by multi-qubit `mz`/`mx`/`my` and passed by `const` reference.
class measure_vector {
  const measure_result *data_;
  std::size_t size_;

  friend struct detail::MeasureResultShim;

  measure_vector(const measure_result *d, std::size_t n) : data_(d), size_(n) {}

public:
  measure_vector() = delete;
  measure_vector(const measure_vector &) = delete;
  measure_vector(measure_vector &&) = delete;
  measure_vector &operator=(const measure_vector &) = delete;
  measure_vector &operator=(measure_vector &&) = delete;

  std::size_t size() const { return size_; }
  const measure_result &operator[](std::size_t idx) const { return data_[idx]; }

  const measure_result *begin() const { return data_; }
  const measure_result *end() const { return data_ + size_; }
};

} // namespace cudaq
