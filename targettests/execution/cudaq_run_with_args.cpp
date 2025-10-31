/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %cpp_std %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

struct Foo {
  bool boolVal;
  std::int64_t i64Val;
  double f64Val;
};

__qpu__ std::vector<bool> kernel_with_struct_args(Foo f) {
  cudaq::qvector q(3);
  if (f.boolVal)
    x(q[0]);
  if (f.i64Val % 2 == 0)
    x(q[1]);
  if (f.f64Val > 5.0)
    x(q[2]);
#ifdef CUDAQ_LIBRARY_MODE
    return cudaq::measure_result::to_bool_vector(mz(q));
#else
    return mz(q);
#endif
}

__qpu__ int kernel_takes_vec_of_vec(std::vector<std::vector<int>> vecOfVec) {
  int sum = 0;
  for (std::size_t i = 0; i < vecOfVec.size(); i++) {
    for (std::size_t j = 0; j < vecOfVec[i].size(); j++) {
      sum += vecOfVec[i][j];
    }
  }
  return sum;
}

struct Baz {
  int multiplier;
  std::vector<float> angles;
};

__qpu__ std::vector<float> kernel_with_struct_having_vec(Baz b) {
  std::size_t n = b.angles.size();
  std::vector<float> results(n);
  for (std::size_t i = 0; i < n; i++) {
    results[i] = b.angles[i] * b.multiplier;
  }
  return results;
}

__qpu__ std::vector<bool>
kernel_with_many_args(int N, bool flag, float angle,
                      std::vector<std::size_t> layers,
                      std::vector<double> parameters,
                      std::vector<std::vector<float>> recursiveVec, Foo var) {
  cudaq::qvector q(N);
  for (std::size_t i = 0; i < q.size(); i++) {
    if (flag)
      h(q[i]);
    if (i < parameters.size() && parameters[i] > 0.5)
      x(q[i]);
  }
  for (std::size_t i = 0; i < layers.size(); i++) {
    if (layers[i] < q.size())
      rz(angle, q[layers[i]]);
  }
  for (std::size_t i = 0; i < recursiveVec.size(); i++) {
    for (std::size_t j = 0; j < recursiveVec[i].size(); j++) {
      if (recursiveVec[i][j] < q.size())
        z(q[recursiveVec[i][j]]);
    }
  }
  if (var.boolVal)
    x(q[N - 1]);
#ifdef CUDAQ_LIBRARY_MODE
    return cudaq::measure_result::to_bool_vector(mz(q));
#else
    return mz(q);
#endif
}

int main() {
  std::size_t shots = 4;
  {
    Foo f = {true, 4, 6.5};
    const auto results = cudaq::run(shots, kernel_with_struct_args, f);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      int c = 0;
      for (const auto &r : results) {
        printf("%d: {", c++);
        for (std::size_t i = 0; i < r.size(); i++)
          printf("%d ", (bool)r[i]);
        printf("}\n");
      }
      printf("success - kernel_with_struct_args\n");
    }
  }
  {
    std::vector<std::vector<int>> vecOfVec = {{1, 2, 3}, {4, 5}, {6}};
    const auto results = cudaq::run(shots, kernel_takes_vec_of_vec, vecOfVec);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      int c = 0;
      for (const auto &r : results) {
        printf("%d: %d\n", c++, r);
      }
      printf("success - kernel_takes_vec_of_vec\n");
    }
  }

  {
    Baz b;
    b.multiplier = 4;
    b.angles = {1.0f, 3.1416f, 2.887f};
    const auto results = cudaq::run(shots, kernel_with_struct_having_vec, b);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      int c = 0;
      for (const auto &r : results) {
        printf("%d: ", c++);
        for (const auto &val : r)
          printf("%f ", val);
        printf("\n");
      }
      printf("success - kernel_with_struct_having_vec\n");
    }
  }

  {
    int N = 5;
    bool flag = true;
    float angle = 1.5708f;
    std::vector<std::size_t> layers = {0, 2, 4};
    std::vector<double> parameters = {0.1, 0.6, 0.3, 0.8, 0.0};
    std::vector<std::vector<float>> recursiveVec = {{1, 3}, {0, 4}};
    Foo var = {true, 3, 4.5};

    const auto results =
        cudaq::run(shots, kernel_with_many_args, N, flag, angle, layers,
                   parameters, recursiveVec, var);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      int c = 0;
      for (const auto &r : results) {
        printf("%d: {", c++);
        for (std::size_t i = 0; i < r.size(); i++)
          printf("%d ", (bool)r[i]);
        printf("}\n");
      }
      printf("success - kernel_with_many_args\n");
    }
  }

  return 0;
}

// CHECK: success - kernel_with_struct_args
// CHECK: success - kernel_takes_vec_of_vec
// CHECK: success - kernel_with_struct_having_vec
// CHECK: success - kernel_with_many_args
