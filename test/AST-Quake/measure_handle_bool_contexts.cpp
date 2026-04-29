/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct WhileCond {
  void operator()() __qpu__ {
    cudaq::qubit q;
    while (mz(q))
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__WhileCond()
// CHECK:           cc.loop while {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.condition %[[VAL_B]]
// CHECK:           } do {
// CHECK:             quake.x

struct ForCond {
  void operator()() __qpu__ {
    cudaq::qubit q;
    for (int i = 0; mz(q); ++i)
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ForCond()
// CHECK:           cc.loop while {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.condition %[[VAL_B]]


struct LogicalNot {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    return !mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__LogicalNot() -> i1
// CHECK:           %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NOT:.*]] = arith.cmpi eq, %[[VAL_B]], %[[VAL_FALSE]] : i1
// CHECK:           return %[[VAL_NOT]] : i1


struct HandleOr {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    return mz(q) || mz(r);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleOr() -> i1
// CHECK:           %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NZ:.*]] = arith.cmpi ne, %[[VAL_B1]], %[[VAL_FALSE]] : i1
// CHECK:           %[[VAL_R:.*]] = cc.if(%[[VAL_NZ]]) -> i1 {
// CHECK:             cc.continue %[[VAL_NZ]] : i1
// CHECK:           } else {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B2:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[VAL_B2]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_R]] : i1


struct BoolInit {
  void operator()() __qpu__ {
    cudaq::qubit q;
    bool b = mz(q);
    if (b)
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__BoolInit()
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_S:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_B]], %[[VAL_S]] : !cc.ptr<i1>


struct HandleNotEqual {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    auto h1 = mz(q);
    auto h2 = mz(r);
    return h1 != h2;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleNotEqual() -> i1
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_M2:.*]] = quake.mz %{{.*}} name "h2" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_B2:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NE:.*]] = arith.cmpi ne, %[[VAL_B1]], %[[VAL_B2]] : i1
// CHECK:           return %[[VAL_NE]] : i1
