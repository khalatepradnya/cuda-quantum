/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include <cmath>
#include <sstream>

inline std::complex<double> &access(std::complex<double> *p,
                                    cudaq::matrix_2::Dimensions sizes,
                                    std::size_t row, std::size_t col) {
  return p[row * sizes.second + col];
}

cudaq::matrix_2 &cudaq::matrix_2::operator*=(const cudaq::matrix_2 &right) {
  if (get_columns() != right.get_rows())
    throw std::runtime_error("matrix dimensions mismatch in operator*=");

  auto new_data = new std::complex<double>[get_rows() * right.get_columns()];
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < right.get_columns(); j++)
      for (std::size_t k = 0; k < get_columns(); k++)
        access(new_data, right.dimensions, i, j) +=
            access(data, dimensions, i, k) *
            access(right.data, right.dimensions, k, j);
  swap(new_data);
  return *this;
}

cudaq::matrix_2 cudaq::operator*(std::complex<double> scalar,
                                 const cudaq::matrix_2 &right) {
  auto new_data =
      new std::complex<double>[right.get_rows() * right.get_columns()];
  for (std::size_t i = 0; i < right.get_rows(); i++)
    for (std::size_t j = 0; j < right.get_columns(); j++)
      access(new_data, right.dimensions, i, j) =
          scalar * access(right.data, right.dimensions, i, j);
  return {new_data, right.dimensions};
}

cudaq::matrix_2 &cudaq::matrix_2::operator+=(const cudaq::matrix_2 &right) {
  if (!(get_rows() == right.get_rows() && get_columns() == right.get_columns()))
    throw std::runtime_error("matrix dimensions mismatch in operator+=");

  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < get_columns(); j++)
      access(data, dimensions, i, j) +=
          access(right.data, right.dimensions, i, j);
  return *this;
}

cudaq::matrix_2 &cudaq::matrix_2::operator-=(const cudaq::matrix_2 &right) {
  if (!(get_rows() == right.get_rows() && get_columns() == right.get_columns()))
    throw std::runtime_error("matrix dimensions mismatch in operator-=");

  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < get_columns(); j++)
      access(data, dimensions, i, j) -=
          access(right.data, right.dimensions, i, j);
  return *this;
}

bool cudaq::operator==(const cudaq::matrix_2 &lhs, const cudaq::matrix_2 &rhs) {
  if (lhs.get_rows() != rhs.get_rows() ||
      lhs.get_columns() != rhs.get_columns()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.get_rows(); i++) {
    for (std::size_t j = 0; j < lhs.get_columns(); j++) {
      if (lhs[{i, j}] != rhs[{i, j}]) {
        return false;
      }
    }
  }

  return true;
}

cudaq::matrix_2 &
cudaq::matrix_2::kronecker_inplace(const cudaq::matrix_2 &right) {
  Dimensions new_dim{get_rows() * right.get_rows(),
                     get_columns() * right.get_columns()};
  auto new_data = new std::complex<double>[get_rows() * right.get_rows() *
                                           get_columns() * right.get_columns()];
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t k = 0; k < right.get_rows(); k++)
      for (std::size_t j = 0; j < get_columns(); j++)
        for (std::size_t m = 0; m < right.get_columns(); m++)
          access(new_data, new_dim, right.get_rows() * i + k,
                 right.get_columns() * j + m) =
              access(data, dimensions, i, j) *
              access(right.data, right.dimensions, k, m);
  swap(new_data);
  dimensions = new_dim;
  return *this;
}

void cudaq::matrix_2::check_size(std::size_t size, const Dimensions &dim) {
  if (size < get_size(dim))
    throw std::runtime_error("vector must have enough elements");
}

std::complex<double>
cudaq::matrix_2::operator[](const std::vector<std::size_t> &at) const {
  if (at.size() != 2)
    throw std::runtime_error("Invalid access: indices must have length of 2");

  if (at[0] >= get_rows() || at[1] >= get_columns())
    throw std::runtime_error(
        "Invalid access: indices {" + std::to_string(at[0]) + ", " +
        std::to_string(at[1]) + "} are larger than matrix dimensions: {" +
        std::to_string(dimensions.first) + ", " +
        std::to_string(dimensions.second) + "}");
  return access(data, dimensions, at[0], at[1]);
}

std::complex<double> &
cudaq::matrix_2::operator[](const std::vector<std::size_t> &at) {
  if (at.size() != 2)
    throw std::runtime_error("Invalid access: indices must have length of 2");

  if (at[0] >= get_rows() || at[1] >= get_columns())
    throw std::runtime_error(
        "Invalid access: indices {" + std::to_string(at[0]) + ", " +
        std::to_string(at[1]) + "} are larger than matrix dimensions: {" +
        std::to_string(dimensions.first) + ", " +
        std::to_string(dimensions.second) + "}");
  return access(data, dimensions, at[0], at[1]);
}

std::string cudaq::matrix_2::dump() const {
  std::ostringstream out;
  out << '{';
  for (std::size_t i = 0; i < get_rows(); i++) {
    out << "  {";
    for (std::size_t j = 0; j < get_columns(); j++)
      out << ' ' << access(data, dimensions, i, j) << ' ';
    out << "}\n ";
  }
  out << '}';
  return out.str();
}

// Calculate the power of a given matrix, `powers` times.
cudaq::matrix_2 cudaq::matrix_2::power(int powers) {
  // Initialize as identity.
  std::size_t rows = get_rows();
  std::size_t columns = get_columns();
  if (rows != columns)
    throw std::runtime_error("Matrix power expects a square matrix.");
  auto result = cudaq::matrix_2(rows, columns);
  for (std::size_t i = 0; i < rows; i++) {
    result[{i, i}] = 1.0;
  }

  // Calculate the matrix power iteratively.
  for (std::size_t i = 0; i < powers; i++) {
    result = result * *this;
  }
  return result;
}

// Calculate the Taylor approximation to the exponential of the given matrix.
cudaq::matrix_2 cudaq::matrix_2::exponential() {
  auto factorial = [](std::size_t value) {
    std::size_t res = 1;
    for (std::size_t factor = 2; factor <= value; ++factor)
      res *= factor;
    return (double)res;
  };

  std::size_t rows = get_rows();
  std::size_t columns = get_columns();
  if (rows != columns)
    throw std::runtime_error("Matrix exponential expects a square matrix.");
  auto result = cudaq::matrix_2(rows, columns);
  // Taylor Series Approximation, fixed at 20 steps.
  std::size_t taylor_steps = 20;
  for (std::size_t step = 0; step < taylor_steps; step++) {
    auto term = this->power(step);
    for (std::size_t i = 0; i < rows; i++) {
      for (std::size_t j = 0; j < columns; j++) {
        result[{i, j}] += term[{i, j}] / factorial(step);
      }
    }
  }
  return result;
}

cudaq::matrix_2 cudaq::matrix_2::identity(const std::size_t rows) {
  auto result = cudaq::matrix_2(rows, rows);
  for (std::size_t i = 0; i < rows; i++)
    result[{i, i}] = 1.;
  return result;
}

// Transpose + Conjugate
cudaq::matrix_2 cudaq::matrix_2::adjoint() {
  std::size_t rows = get_rows();
  std::size_t cols = get_columns();
  matrix_2 result(cols, rows);

  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      result[{j, i}] = std::conj((*this)[{i, j}]);
    }
  }

  return result;
}