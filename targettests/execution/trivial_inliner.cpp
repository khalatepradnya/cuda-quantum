/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s

#include <cudaq.h>

struct MyTuple {
  bool boolVal;
  std::int64_t i64Val;
  double f64Val;
};

auto struct_test = [](MyTuple t) __qpu__ { return t; };
auto simple_test = []() __qpu__ { return 42; };
auto int_test = [](int i) __qpu__ { return i; };
auto float_test = [](float f) __qpu__ { return f; };
auto double_test = [](double d) __qpu__ { return d; };
auto bool_test = [](bool b) __qpu__ { return b; };

struct empty {};

auto kernel = []() __qpu__ { return empty{}; };

auto kernel_with_args = [](int num_qubits) __qpu__ {
  cudaq::qvector q(num_qubits);
  h(q);
  return empty{};
};

__qpu__ std::int8_t simple_int8() {
  return 13;
}

__qpu__ std::int16_t simple_int16() {
  return 32000;
}

__qpu__ std::int32_t simple_int32() {
  return 2000000000;
}

__qpu__ std::int64_t simple_int64() {
  return 9000000000000000000;
}

__qpu__ std::size_t simple_size_t() {
  return 123456789;
}

__qpu__ std::int8_t incrementer_int8(std::int8_t val) {
  return val + 1;
}

__qpu__ std::int64_t incrementer_int64(std::int64_t val) {
  return val + 1;
}

int main() {
  int c = 0;
  {
    MyTuple t = {true, 654, 9.123};
    const auto results = cudaq::run(3, struct_test, t);
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: {%s, %ld, %.3f}\n", c++, i.boolVal ? "true" : "false",
               i.i64Val, i.f64Val);
      printf("success!\n");
    }
  }
  {
    const auto results = cudaq::run(5, simple_test);
    if (results.size() != 5) {
      printf("FAILED! Expected 5 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }
  {
    const auto results = cudaq::run(4, int_test, 37);
    if (results.size() != 4) {
      printf("FAILED! Expected 4 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }
  {
    const auto results = cudaq::run(3, float_test, 2.71828f);
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: %.3f\n", c++, i);
      printf("success!\n");
    }
  }
  {
    const auto results = cudaq::run(2, double_test, 1.41421356);
    if (results.size() != 2) {
      printf("FAILED! Expected 2 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: %.4f\n", c++, i);
      printf("success!\n");
    }
  }
  {
    const auto results = cudaq::run(5, bool_test, true);
    if (results.size() != 5) {
      printf("FAILED! Expected 5 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: %s\n", c++, i ? "true" : "false");
      printf("success - bool_test\n");
    }
  }
  {
    auto results = cudaq::run(5, kernel);
    // Test here is that this does not crash.
    printf("success - kernel returning empty struct\n");
  }
  {
    auto results = cudaq::run(1, kernel_with_args, 4);
    // Test here is that this does not crash.
    printf("success - kernel_with_args\n");
  }
  std::size_t shots = 2;
  {
    const auto results = cudaq::run(shots, simple_int8);
    c = 0;
    for (auto i : results)
      printf("%d: %d\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, simple_int16);
    c = 0;
    for (auto i : results)
      printf("%d: %d\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, simple_int32);
    c = 0;
    for (auto i : results)
      printf("%d: %d\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, simple_int64);
    c = 0;
    for (auto i : results)
      printf("%d: %ld\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, simple_size_t);
    c = 0;
    for (auto i : results)
      printf("%d: %zu\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, incrementer_int8, 41);
    c = 0;
    for (auto i : results)
      printf("%d: %d\n", c++, i);
  }
  {
    const auto results = cudaq::run(shots, incrementer_int64, 90000000000000000);
    c = 0;
    for (auto i : results)
      printf("%d: %ld\n", c++, i);
  }
  return 0;
}

// CHECK: 0: {true, 654, 9.123}
// CHECK: 1: {true, 654, 9.123}
// CHECK: 2: {true, 654, 9.123}
// CHECK: success!
// CHECK: 0: 42
// CHECK: 1: 42
// CHECK: 2: 42
// CHECK: 3: 42
// CHECK: 4: 42
// CHECK: success!
// CHECK: 0: 37
// CHECK: 1: 37
// CHECK: 2: 37
// CHECK: 3: 37
// CHECK: success!
// CHECK: 0: 2.718
// CHECK: 1: 2.718
// CHECK: 2: 2.718
// CHECK: success!
// CHECK: 0: 1.4142
// CHECK: 1: 1.4142
// CHECK: success!
// CHECK: 0: true
// CHECK: 1: true
// CHECK: 2: true
// CHECK: 3: true
// CHECK: 4: true
// CHECK: success - bool_test
// CHECK: success - kernel returning empty struct
// CHECK: success - kernel_with_args
// CHECK: 0: 13
// CHECK: 1: 13
// CHECK: 0: 32000
// CHECK: 1: 32000
// CHECK: 0: 2000000000
// CHECK: 1: 2000000000
// CHECK: 0: 9000000000000000000
// CHECK: 1: 9000000000000000000
// CHECK: 0: 123456789
// CHECK: 1: 123456789
// CHECK: 0: 42
// CHECK: 1: 42
// CHECK: 0: 90000000000000001
// CHECK: 1: 90000000000000001
