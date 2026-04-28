/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <tuple>
#include <utility>
#include <vector>

// expected-note@* 0+ {{}}

struct BoundaryDirectParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle h) __qpu__ { (void)h; }
};

struct BoundaryDirectReturn {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  cudaq::measure_handle operator()() __qpu__ {
    cudaq::qubit q;
    return mz(q);
  }
};

struct BoundaryVectorParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::vector<cudaq::measure_handle> h) __qpu__ { (void)h; }
};

struct BoundaryTupleParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::tuple<int, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPairParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<bool, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPointerParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle *h) __qpu__ { (void)h; }
};

struct BoundaryReferenceParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle &h) __qpu__ { (void)h; }
};

struct MeasureHandleHolder {
  cudaq::measure_handle h;
};

struct BoundaryAggregateParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(MeasureHandleHolder s) __qpu__ { (void)s; }
};

struct BoundaryPairOfVectorParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<int, std::vector<cudaq::measure_handle>> p) __qpu__ {
    (void)p;
  }
};
