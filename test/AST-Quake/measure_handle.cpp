/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s


#include <cudaq.h>

__qpu__ bool consume_handle(cudaq::qubit &q, const cudaq::measure_handle &h) {
  return static_cast<bool>(h);
}

struct CrossFunctionCaller {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    return consume_handle(q, h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_consume_handle.
// CHECK:           %[[VAL_H:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_H]] : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CrossFunctionCaller() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]]{{.*}}: (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_S:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_S]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_R:.*]] = call @__nvqpp__mlirgen__function_consume_handle.{{.*}}(%[[VAL_Q]], %[[VAL_S]]){{.*}}: (!quake.ref, !cc.ptr<!cc.measure_handle>) -> i1
// CHECK:           return %[[VAL_R]] : i1
// CHECK:         }
