// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --phys-to-quake=entry=bit %s | FileCheck %s --check-prefix=BIT --implicit-check-not=phys. --implicit-check-not=qlx.
// RUN: qlx-opt --phys-to-quake=entry=flag %s | FileCheck %s --check-prefix=FLAG --implicit-check-not=phys. --implicit-check-not=qlx.
// RUN: qlx-opt --phys-to-quake=entry=none %s | FileCheck %s --check-prefix=NONE --implicit-check-not=phys. --implicit-check-not=qlx.
// RUN: not qlx-opt --phys-to-quake %s 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not qlx-opt --phys-to-quake=entry=missing %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not qlx-opt --phys-to-quake=entry=machine %s 2>&1 | FileCheck %s --check-prefix=MACHINE

// The pass translates only the graph that `entry` names. It erases the other
// graphs, @unsupported included, and Logical's `qlx.*` module attributes.
// One Boolean result is an `i1`, and a graph without results gives a function
// without one. `entry` must name a phys.graph.
module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @machine {
    phys.resource_class @qubits {count = 3 : i64, kind = "qubit", native_actions = [], native_instruments = [@mz]}
  }
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @b {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @c {architecture = @machine, index = 2 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @bit on @machine : () -> !phys.record<@bit> {
    %0 = phys.acquire [@a] {event_id = "bit0"} : !phys.state<@a>
    %record = phys.measure @mz(%0) {destructive, event_id = "bit1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
    phys.return %record : !phys.record<@bit>
  }
  phys.graph @flag on @machine : (i1) -> i1 {
  ^bb0(%arg0: i1):
    %0 = phys.condition %arg0 {event_id = "flag0", expected = false} : i1 -> i1
    phys.return %0 : i1
  }
  phys.graph @none on @machine : () -> () {
    %0 = phys.acquire [@b] {event_id = "none0"} : !phys.state<@b>
    phys.release %0 {event_id = "none1"} : !phys.state<@b>
    phys.return
  }
  phys.graph @unsupported on @machine : () -> !phys.state<@c> {
    %0 = phys.acquire [@c] {event_id = "unsupported0"} : !phys.state<@c>
    phys.return %0 : !phys.state<@c>
  }
}

// BIT:         module attributes {quake.noOptimization} {
// BIT-LABEL:   func.func @__nvqpp__mlirgen__bit() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// BIT:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// BIT:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[NULL_WIRE_0]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// BIT:           quake.sink %[[MZ_0]] : !quake.wire
// BIT:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
// BIT:           return %[[DISCRIMINATE_0]] : i1
// BIT:         }

// FLAG:         module attributes {quake.noOptimization} {
// FLAG-LABEL:   func.func @__nvqpp__mlirgen__flag(
// FLAG-SAME:      %[[ARG0:.*]]: i1) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// FLAG:           %[[CONSTANT_0:.*]] = arith.constant true
// FLAG:           %[[XORI_0:.*]] = arith.xori %[[ARG0]], %[[CONSTANT_0]] : i1
// FLAG:           return %[[XORI_0]] : i1
// FLAG:         }

// NONE:         module attributes {quake.noOptimization} {
// NONE-LABEL:   func.func @__nvqpp__mlirgen__none() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// NONE:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// NONE:           quake.sink %[[NULL_WIRE_0]] : !quake.wire
// NONE:           return
// NONE:         }

// EMPTY: error: phys-to-quake needs the entry option to name a phys.graph, but '' names none
// MISSING: error: phys-to-quake needs the entry option to name a phys.graph, but 'missing' names none
// MACHINE: error: phys-to-quake needs the entry option to name a phys.graph, but 'machine' names none
