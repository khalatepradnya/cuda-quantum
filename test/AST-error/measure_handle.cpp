/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <functional>
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

// A callable parameter whose inner signature mentions `measure_handle`
// would still cross the boundary at every invocation: the kernel calls
// the host-supplied callable with a handle argument. The boundary
// check must descend into callable signatures to catch this; the plain
// `containsMeasureHandle` is callable-blind by design (see the
// `containsMeasureHandleAtBoundary` doc comment in
// `include/cudaq/Optimizer/Dialect/CC/CCTypes.h`).

struct BoundaryFunctionTypeParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::function<void(cudaq::measure_handle)> f) __qpu__ {
    (void)f;
  }
};

struct BoundaryQkernelParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::qkernel<void(cudaq::measure_handle)> k) __qpu__ {
    (void)k;
  }
};

// Note: free `__qpu__` functions whose signature mentions `measure_handle`
// are not diagnosed here. The spec (§Host-Device Boundary) rejects only
// *entry-point* kernels with a handle in the signature. A free function
// with a handle parameter or return cannot be an entry point (the host
// has no way to synthesize a handle), so the bridge silently demotes it
// to a device-only helper. Positive coverage lives in
// `test/AST-Quake/const_reference_extension.cpp` (handle-vector parameter)
// and `test/AST-Quake/measure_handle.cpp` (handle return).

// Spec (`measure_handle.bs` §C++ API L96): `cudaq::to_integer(mz(qvec))`
// must be migrated explicitly to
// `cudaq::to_integer(cudaq::to_bools(mz(qvec)))`. The bridge rejects the
// direct shape rather than silently inserting a discriminate, so the
// explicit migration cannot regress in user code. Positive coverage of
// the explicit form lives in
// `test/AST-Quake/measure_handle_to_integer.cpp`.

void sink(std::int64_t);

struct ToIntegerDirectRejected {
  void operator()() __qpu__ {
    cudaq::qvector q(8);
    // expected-error@+1{{cudaq::to_integer accepts std::vector<bool>; wrap measurement results with cudaq::to_bools(...) first}}
    sink(cudaq::to_integer(mz(q)));
  }
};
