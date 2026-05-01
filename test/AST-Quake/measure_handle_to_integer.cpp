/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// `cudaq::to_integer` accepts a `std::vector<bool>` by spec, but the bridge
// also accepts the direct `to_integer(mz(qvec))` shape now that `mz` on a
// register returns `std::vector<measure_handle>`. The intrinsic
// `__nvqpp_cudaqConvertToInteger` is typed `(!cc.stdvec<i1>) -> i64`, so
// the bridge inserts a `quake.discriminate` to bridge the type gap before
// the call; without it the verifier rejects the resulting IR.

#include <cudaq.h>

void sink(std::int64_t);

struct ToIntegerDirect {
  void operator()() __qpu__ {
    cudaq::qvector q(8);
    sink(cudaq::to_integer(mz(q)));
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ToIntegerDirect
// CHECK:         %[[MZ:.*]] = quake.mz %{{.*}} : (!quake.veq<8>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:         %[[BOOLS:.*]] = quake.discriminate %[[MZ]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK:         %{{.*}} = call @__nvqpp_cudaqConvertToInteger(%[[BOOLS]]) : (!cc.stdvec<i1>) -> i64

// The explicit `to_integer(to_bools(mz(qvec)))` form still works -- the
// `to_bools` already produces `!cc.stdvec<i1>`, so the bridge does not
// emit a redundant discriminate.
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
