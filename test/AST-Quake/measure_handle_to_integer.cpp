/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// `cudaq::to_integer` accepts a `std::vector<bool>` by spec. Per
// `cudaq-spec/proposals/measure_handle.bs` §C++ API L96, calling
// `cudaq::to_integer(mz(qvec))` directly on a measurement result is
// rejected by the bridge -- users must explicitly migrate to
// `cudaq::to_integer(cudaq::to_bools(mz(qvec)))`. This test locks the
// explicit form's IR shape; the negative path (rejection of the direct
// form) lives in `test/AST-error/measure_handle.cpp`.

#include <cudaq.h>

void sink(std::int64_t);

// The explicit `to_integer(to_bools(mz(qvec)))` form: `to_bools` already
// produces `!cc.stdvec<i1>`, so the bridge does not insert a redundant
// discriminate before the `__nvqpp_cudaqConvertToInteger` call.
struct ToIntegerExplicit {
  void operator()() __qpu__ {
    cudaq::qvector q(8);
    sink(cudaq::to_integer(cudaq::to_bools(mz(q))));
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ToIntegerExplicit
// CHECK:         %[[BOOLS:.*]] = quake.discriminate %{{.*}} : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK-NOT:     quake.discriminate
// CHECK:         %{{.*}} = call @__nvqpp_cudaqConvertToInteger(%[[BOOLS]]) : (!cc.stdvec<i1>) -> i64
