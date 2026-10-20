// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --split-input-file --phys-to-quake=entry=entry %s | FileCheck %s --implicit-check-not=phys. --implicit-check-not=qlx.

// Carriers are allocated and freed in program order: @b reuses index 0 only
// after @a's destructive measurement. Every prepared or reset state is a
// reset, plus `h` for the plus states. The machine's @mx, whose process is
// `measure_x`, shadows the module's @mx, so the retained measurement is
// `h; mz; h`.
module {
  phys.machine @machine {
    phys.instrument @mx {arity = 1 : i64, kind = "measure_x", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_x\",\"parameters\":{}}", record_schema = "bit"}
    phys.resource_class @qubits {count = 1 : i64, kind = "qubit", native_actions = [@x], native_instruments = [@mz, @mx]}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.instrument @mx {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @b {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @machine : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.prepare %0 {event_id = "e1", state = "zero"} : (!phys.state<@a>) -> !phys.state<@a>
    %2 = phys.prepare %1 {event_id = "e2", state = "0"} : (!phys.state<@a>) -> !phys.state<@a>
    %3 = phys.prepare %2 {event_id = "e3", state = "loaded"} : (!phys.state<@a>) -> !phys.state<@a>
    %4 = phys.prepare %3 {event_id = "e4", state = "plus"} : (!phys.state<@a>) -> !phys.state<@a>
    %5 = phys.prepare %4 {event_id = "e5", state = "+"} : (!phys.state<@a>) -> !phys.state<@a>
    %6 = phys.reset %5 {event_id = "e6"} : (!phys.state<@a>) -> !phys.state<@a>
    %7 = phys.reset %6 {event_id = "e7", state = "plus"} : (!phys.state<@a>) -> !phys.state<@a>
    %8, %r0 = phys.measure @mz(%7) {event_id = "e8", record_id = "r0"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %9, %r1 = phys.measure @mx(%8) {event_id = "e9", record_id = "r1"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %r2 = phys.measure @mz(%9) {destructive, event_id = "e10", record_id = "r2"} : (!phys.state<@a>) -> !phys.record<@bit>
    %10 = phys.acquire [@b] {event_id = "e11"} : !phys.state<@b>
    %11 = phys.apply @x(%10) {event_id = "e12"} : (!phys.state<@b>) -> !phys.state<@b>
    phys.release %11 {event_id = "e13"} : !phys.state<@b>
    phys.return
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// CHECK:           %[[RESET_0:.*]] = quake.reset %[[NULL_WIRE_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_1:.*]] = quake.reset %[[RESET_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_2:.*]] = quake.reset %[[RESET_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_3:.*]] = quake.reset %[[RESET_2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_0:.*]] = quake.h %[[RESET_3]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_4:.*]] = quake.reset %[[H_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_1:.*]] = quake.h %[[RESET_4]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_5:.*]] = quake.reset %[[H_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_6:.*]] = quake.reset %[[RESET_5]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_2:.*]] = quake.h %[[RESET_6]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[H_2]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[H_3:.*]] = quake.h %[[MZ_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_1:.*]], %[[MZ_1:.*]] = quake.mz %[[H_3]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[H_4:.*]] = quake.h %[[MZ_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_2:.*]], %[[MZ_2:.*]] = quake.mz %[[H_4]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           quake.sink %[[MZ_2]] : !quake.wire
// CHECK:           %[[NULL_WIRE_1:.*]] = quake.null_wire
// CHECK:           %[[X_0:.*]] = quake.x %[[NULL_WIRE_1]] : (!quake.wire) -> !quake.wire
// CHECK:           quake.sink %[[X_0]] : !quake.wire
// CHECK:           return
// CHECK:         }

// -----

// The process names the gate, not the symbol: @x runs `h`, and the machine's
// @flip (`x`) shadows the module's @flip (`z`). `sdg` and `tdg` are
// adjoints, and `cx` and `cz` take operand 0 as the control. A broadcast
// action applies to each operand, and batched lanes are consecutive groups.
module {
  phys.machine @machine {
    phys.action @flip {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
    phys.resource_class @qubits {count = 4 : i64, kind = "qubit", native_actions = [@x, @flip, @s, @sdg, @t, @tdg, @y, @z, @reset, @cx, @cz, @swap, @global_h], native_instruments = []}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}"}
  phys.action @flip {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"z\",\"parameters\":{}}"}
  phys.action @s {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"s\",\"parameters\":{}}"}
  phys.action @sdg {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"sdg\",\"parameters\":{}}"}
  phys.action @t {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"t\",\"parameters\":{}}"}
  phys.action @tdg {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"tdg\",\"parameters\":{}}"}
  phys.action @y {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"y\",\"parameters\":{}}"}
  phys.action @z {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"z\",\"parameters\":{}}"}
  phys.action @reset {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"reset\",\"parameters\":{}}"}
  phys.action @cx {arity = 2 : i64, process = "{\"kind\":\"builtin\",\"name\":\"cx\",\"parameters\":{}}"}
  phys.action @cz {arity = 2 : i64, process = "{\"kind\":\"builtin\",\"name\":\"cz\",\"parameters\":{}}"}
  phys.action @swap {arity = 2 : i64, process = "{\"kind\":\"builtin\",\"name\":\"swap\",\"parameters\":{}}"}
  phys.action @global_h {arity = 1 : i64, broadcast, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}"}
  phys.resource @c0 {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @t0 {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @c1 {architecture = @machine, index = 2 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @t1 {architecture = @machine, index = 3 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @machine : () -> () {
    %0:4 = phys.acquire [@c0, @t0, @c1, @t1] {event_id = "e0"} : !phys.state<@c0>, !phys.state<@t0>, !phys.state<@c1>, !phys.state<@t1>
    %1 = phys.apply @x(%0#0) {event_id = "e1"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %2 = phys.apply @flip(%1) {event_id = "e2"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %3 = phys.apply @s(%2) {event_id = "e3"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %4 = phys.apply @sdg(%3) {event_id = "e4"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %5 = phys.apply @t(%4) {event_id = "e5"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %6 = phys.apply @tdg(%5) {event_id = "e6"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %7 = phys.apply @y(%6) {event_id = "e7"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %8 = phys.apply @z(%7) {event_id = "e8"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %9 = phys.apply @reset(%8) {event_id = "e9"} : (!phys.state<@c0>) -> !phys.state<@c0>
    %10:2 = phys.apply @cz(%0#1, %9) {event_id = "e10"} : (!phys.state<@t0>, !phys.state<@c0>) -> (!phys.state<@t0>, !phys.state<@c0>)
    %11:2 = phys.apply @swap(%10#0, %10#1) {event_id = "e11"} : (!phys.state<@t0>, !phys.state<@c0>) -> (!phys.state<@t0>, !phys.state<@c0>)
    %12:2 = phys.apply @global_h(%11#1, %0#2) {event_id = "e12"} : (!phys.state<@c0>, !phys.state<@c1>) -> (!phys.state<@c0>, !phys.state<@c1>)
    %13:4 = phys.apply @cx(%12#0, %11#0, %12#1, %0#3) {batch_lanes = 2 : i64, event_id = "e13"} : (!phys.state<@c0>, !phys.state<@t0>, !phys.state<@c1>, !phys.state<@t1>) -> (!phys.state<@c0>, !phys.state<@t0>, !phys.state<@c1>, !phys.state<@t1>)
    phys.release %13#0, %13#1, %13#2, %13#3 {event_id = "e14"} : !phys.state<@c0>, !phys.state<@t0>, !phys.state<@c1>, !phys.state<@t1>
    phys.return
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_1:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_2:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_3:.*]] = quake.null_wire
// CHECK:           %[[H_0:.*]] = quake.h %[[NULL_WIRE_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_0:.*]] = quake.x %[[H_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_0:.*]] = quake.s %[[X_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_1:.*]] = quake.s<adj> %[[S_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[T_0:.*]] = quake.t %[[S_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[T_1:.*]] = quake.t<adj> %[[T_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[Y_0:.*]] = quake.y %[[T_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[Z_0:.*]] = quake.z %[[Y_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[RESET_0:.*]] = quake.reset %[[Z_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[Z_1:.*]]:2 = quake.z {{\[}}%[[NULL_WIRE_1]]] %[[RESET_0]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[SWAP_0:.*]]:2 = quake.swap %[[Z_1]]#0, %[[Z_1]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[H_1:.*]] = quake.h %[[SWAP_0]]#1 : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_2:.*]] = quake.h %[[NULL_WIRE_2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_1:.*]]:2 = quake.x {{\[}}%[[H_1]]] %[[SWAP_0]]#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X_2:.*]]:2 = quake.x {{\[}}%[[H_2]]] %[[NULL_WIRE_3]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           quake.sink %[[X_1]]#0 : !quake.wire
// CHECK:           quake.sink %[[X_1]]#1 : !quake.wire
// CHECK:           quake.sink %[[X_2]]#0 : !quake.wire
// CHECK:           quake.sink %[[X_2]]#1 : !quake.wire
// CHECK:           return
// CHECK:         }

// -----

// A record of any schema stays a measurement handle until it is first read
// and is discriminated once. `condition` is `bit == expected`, and `xor` and
// `all_false` are `i1` logic, also with no events. `arith` ops are cloned,
// and an `index` input is an `i64` parameter with a signed cast. Delay,
// barrier, move, and fence emit nothing. Several results form one list in
// return order.
module {
  phys.machine @machine {
    phys.resource_class @qubits {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@mz]}
  }
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "outcome"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @machine : (index, f64, i1) -> (!phys.record<@outcome>, i1, i1, i1, i1, i1, i1) {
  ^bb0(%arg0: index, %arg1: f64, %arg2: i1):
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.delay %0 {duration_ns = 2.000000e+01 : f64, event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    %2 = phys.barrier %1 {event_id = "e2"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.barrier  {domains = ["clock"], event_id = "e3"} : () -> ()
    %3 = phys.move @lane(%2) {event_id = "e4", trajectory = "shuttle"} : (!phys.state<@a>) -> !phys.state<@a>
    event.fence {effects = ["quantum"], event_id = "e5"}
    %record = phys.measure @mz(%3) {destructive, event_id = "e6", record_id = "r0"} : (!phys.state<@a>) -> !phys.record<@outcome>
    %4 = phys.condition %record {event_id = "e7"} : !phys.record<@outcome> -> i1
    %5 = phys.condition %record {event_id = "e8", expected = false} : !phys.record<@outcome> -> i1
    %6 = phys.condition %arg2 {event_id = "e9", expected = false} : i1 -> i1
    %7 = phys.xor %4, %6 {event_id = "e10"} : i1
    %8 = phys.all_false {event_id = "e11"} : () -> i1
    %9 = phys.all_false %5, %7 {event_id = "e12"} : (i1, i1) -> i1
    %c3 = arith.constant 3 : index
    %10 = arith.cmpi slt, %arg0, %c3 : index
    %11 = arith.mulf %arg1, %arg1 : f64
    %cst = arith.constant 2.500000e-01 : f64
    %12 = arith.cmpf ogt, %11, %cst : f64
    phys.return %record, %5, %7, %8, %9, %10, %12 : !phys.record<@outcome>, i1, i1, i1, i1, i1, i1
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry(
// CHECK-SAME:      %[[ARG0:.*]]: i64,
// CHECK-SAME:      %[[ARG1:.*]]: f64,
// CHECK-SAME:      %[[ARG2:.*]]: i1) -> !cc.sequence<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG0]] : i64 to index
// CHECK:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// CHECK:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[NULL_WIRE_0]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           quake.sink %[[MZ_0]] : !quake.wire
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[DISCRIMINATE_0]], %[[CONSTANT_0]] : i1
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[XORI_1:.*]] = arith.xori %[[ARG2]], %[[CONSTANT_1]] : i1
// CHECK:           %[[XORI_2:.*]] = arith.xori %[[DISCRIMINATE_0]], %[[XORI_1]] : i1
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant true
// CHECK:           %[[XORI_3:.*]] = arith.xori %[[CONSTANT_2]], %[[CONSTANT_3]] : i1
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant false
// CHECK:           %[[ORI_0:.*]] = arith.ori %[[CONSTANT_4]], %[[XORI_0]] : i1
// CHECK:           %[[ORI_1:.*]] = arith.ori %[[ORI_0]], %[[XORI_2]] : i1
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
// CHECK:           %[[XORI_4:.*]] = arith.xori %[[ORI_1]], %[[CONSTANT_5]] : i1
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[INDEX_CAST_0]], %[[CONSTANT_6]] : index
// CHECK:           %[[MULF_0:.*]] = arith.mulf %[[ARG1]], %[[ARG1]] : f64
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 2.500000e-01 : f64
// CHECK:           %[[CMPF_0:.*]] = arith.cmpf ogt, %[[MULF_0]], %[[CONSTANT_7]] : f64
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca !cc.array<i1 x 7>
// CHECK:           %[[COMPUTE_PTR_0:.*]] = cc.compute_ptr %[[ALLOCA_0]][0] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[DISCRIMINATE_0]], %[[COMPUTE_PTR_0]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_1:.*]] = cc.compute_ptr %[[ALLOCA_0]][1] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[XORI_0]], %[[COMPUTE_PTR_1]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_2:.*]] = cc.compute_ptr %[[ALLOCA_0]][2] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[XORI_2]], %[[COMPUTE_PTR_2]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_3:.*]] = cc.compute_ptr %[[ALLOCA_0]][3] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[XORI_3]], %[[COMPUTE_PTR_3]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_4:.*]] = cc.compute_ptr %[[ALLOCA_0]][4] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[XORI_4]], %[[COMPUTE_PTR_4]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_5:.*]] = cc.compute_ptr %[[ALLOCA_0]][5] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[CMPI_0]], %[[COMPUTE_PTR_5]] : !cc.ptr<i1>
// CHECK:           %[[COMPUTE_PTR_6:.*]] = cc.compute_ptr %[[ALLOCA_0]][6] : (!cc.ptr<!cc.array<i1 x 7>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[CMPF_0]], %[[COMPUTE_PTR_6]] : !cc.ptr<i1>
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 7 : i64
// CHECK:           %[[SEQUENCE_INIT_0:.*]] = cc.sequence_init %[[ALLOCA_0]], %[[CONSTANT_8]] : (!cc.ptr<!cc.array<i1 x 7>>, i64) -> !cc.sequence<i1>
// CHECK:           %[[SEQUENCE_DATA_0:.*]] = cc.sequence_data %[[SEQUENCE_INIT_0]] : (!cc.sequence<i1>) -> !cc.ptr<i8>
// CHECK:           %[[SEQUENCE_SIZE_0:.*]] = cc.sequence_size %[[SEQUENCE_INIT_0]] : (!cc.sequence<i1>) -> i64
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = call @__nvqpp_vectorCopyCtor(%[[SEQUENCE_DATA_0]], %[[SEQUENCE_SIZE_0]], %[[CONSTANT_9]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[SEQUENCE_INIT_1:.*]] = cc.sequence_init %[[VAL_1]], %[[SEQUENCE_SIZE_0]] : (!cc.ptr<i8>, i64) -> !cc.sequence<i1>
// CHECK:           return %[[SEQUENCE_INIT_1]] : !cc.sequence<i1>
// CHECK:         }

// -----

// A call is inlined. A template replays the call that `template_event` names,
// with `state_aliases` renaming the body's carriers: "flip_b" flips @b, and
// "measure_b" measures it. The elided "flip_b" has no SSA boundary, so %1
// still names @b after it. "flip_c", inside the call "outer", replays "flip"
// on @c, and the body of "outer" captures the records of the calls before it.
module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @gate_spec for @objective : () -> () {encodings = []}
  fabric.gadget @gate() {
    fabric.return
  } {realization_boundary = {}, spec = @gate_spec}
  phys.machine @m {
    phys.resource_class @q {count = 3 : i64, kind = "qubit", native_actions = [@x, @h], native_instruments = [@mz]}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.action @h {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.resource @c {index = 2 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> (!phys.record<@bit>, !phys.record<@bit>, i1) {
    %0 = phys.acquire [@a] {event_id = "acquire_a"} : !phys.state<@a>
    %1 = phys.acquire [@b] {event_id = "acquire_b"} : !phys.state<@b>
    %2 = "phys.call"(%0) <{callee = @gate, event_id = "flip", instance = "gate.0"}> ({
    ^bb0(%s: !phys.state<@a>):
      %t = phys.apply @x(%s) {event_id = "x"} : (!phys.state<@a>) -> !phys.state<@a>
      phys.yield %t : !phys.state<@a>
    }) : (!phys.state<@a>) -> !phys.state<@a>
    phys.call_template {callee = @gate, event_id = "flip_b", instance = "gate.1", state_aliases = [{alias = @b, template = @a}], state_boundary_elided, template_event = "flip"} : () -> ()
    %3:2 = "phys.call"(%2) <{callee = @gate, event_id = "measure", instance = "gate.2"}> ({
    ^bb0(%s: !phys.state<@a>):
      %o, %r = phys.measure @mz(%s) {event_id = "mz", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
      phys.yield %o, %r : !phys.state<@a>, !phys.record<@bit>
    }) : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %4:2 = phys.call_template %1 {callee = @gate, event_id = "measure_b", instance = "gate.3", state_aliases = [{alias = @b, template = @a}], template_event = "measure"} : (!phys.state<@b>) -> (!phys.state<@b>, !phys.record<@bit>)
    %5 = phys.acquire [@c] {event_id = "acquire_c"} : !phys.state<@c>
    %6:2 = "phys.call"(%5) <{callee = @gate, event_id = "outer", instance = "gate.4"}> ({
    ^bb0(%s: !phys.state<@c>):
      phys.call_template {callee = @gate, event_id = "flip_c", instance = "gate.5", state_aliases = [{alias = @c, template = @a}], state_boundary_elided, template_event = "flip"} : () -> ()
      %p = phys.condition %3#1 {event_id = "bit_a", expected = true} : !phys.record<@bit> -> i1
      %q = phys.condition %4#1 {event_id = "bit_b", expected = true} : !phys.record<@bit> -> i1
      %e = phys.xor %p, %q {event_id = "parity"} : i1
      phys.yield %s, %e : !phys.state<@c>, i1
    }) : (!phys.state<@c>) -> (!phys.state<@c>, i1)
    phys.release %3#0, %4#0, %6#0 {event_id = "release"} : !phys.state<@a>, !phys.state<@b>, !phys.state<@c>
    phys.return %3#1, %4#1, %6#1 : !phys.record<@bit>, !phys.record<@bit>, i1
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry() -> !cc.sequence<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_1:.*]] = quake.null_wire
// CHECK:           %[[X_0:.*]] = quake.x %[[NULL_WIRE_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_1:.*]] = quake.x %[[NULL_WIRE_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[X_0]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[VAL_1:.*]], %[[MZ_1:.*]] = quake.mz %[[X_1]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[NULL_WIRE_2:.*]] = quake.null_wire
// CHECK:           %[[X_2:.*]] = quake.x %[[NULL_WIRE_2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
// CHECK:           %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[DISCRIMINATE_0]], %[[DISCRIMINATE_1]] : i1

// -----

// A product measurement collects the parity on a fresh ancilla. X and Y
// operands are changed to Z around their CX, and `invert` complements the
// bit where it is read. A product rotation is `exp(-i angle P / 2)`: a CX
// ladder around `rz`, or around `s`, `z`, or `sdg` for a multiple of pi/2.
module {
  phys.instrument @mpp {kind = "measure_product", variadic, record_schema = "bit", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_product\",\"parameters\":{}}"}
  phys.machine @m {
    phys.resource_class @q {count = 2 : i64, kind = "qubit", native_actions = [], native_instruments = [@mpp]}
  }
  phys.resource @a {capabilities = ["qlx.physical/native_pauli_product_rotation"], index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {capabilities = ["qlx.physical/native_pauli_product_rotation"], index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> (!phys.record<@bit>, !phys.record<@bit>) {
    %0:2 = phys.acquire [@a, @b] {event_id = "e0"} : !phys.state<@a>, !phys.state<@b>
    %1:3 = phys.measure_product %0#0, %0#1 {event_id = "e1", instrument = @mpp, paulis = ["X", "Y"], record_id = "r0"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>, !phys.record<@bit>)
    %2:3 = phys.measure_product %1#0, %1#1 {event_id = "e2", instrument = @mpp, invert, paulis = ["Z", "Z"], record_id = "r1"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>, !phys.record<@bit>)
    %3:2 = phys.rotate_product %2#0, %2#1 {angle = 1.250000e-01 : f64, event_id = "e3", paulis = ["Y", "Z"]} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>)
    %4 = phys.rotate_product %3#0 {angle = 1.5707963267948966 : f64, event_id = "e4", paulis = ["X"]} : (!phys.state<@a>) -> !phys.state<@a>
    %5 = phys.rotate_product %4 {angle = -3.1415926535897931 : f64, event_id = "e5", paulis = ["Z"]} : (!phys.state<@a>) -> !phys.state<@a>
    %6 = phys.rotate_product %5 {angle = -1.5707963267948966 : f64, event_id = "e6", paulis = ["Z"]} : (!phys.state<@a>) -> !phys.state<@a>
    %7 = phys.rotate_product %6 {angle = 6.2831853071795862 : f64, event_id = "e7", paulis = ["Z"]} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %7, %3#1 {event_id = "e8"} : !phys.state<@a>, !phys.state<@b>
    phys.return %1#2, %2#2 : !phys.record<@bit>, !phys.record<@bit>
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry() -> !cc.sequence<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_1:.*]] = quake.null_wire
// CHECK:           %[[NULL_WIRE_2:.*]] = quake.null_wire
// CHECK:           %[[H_0:.*]] = quake.h %[[NULL_WIRE_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_0:.*]]:2 = quake.x {{\[}}%[[H_0]]] %[[NULL_WIRE_2]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[H_1:.*]] = quake.h %[[X_0]]#0 : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_0:.*]] = quake.s<adj> %[[NULL_WIRE_1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_2:.*]] = quake.h %[[S_0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_1:.*]]:2 = quake.x {{\[}}%[[H_2]]] %[[X_0]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[H_3:.*]] = quake.h %[[X_1]]#0 : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_1:.*]] = quake.s %[[H_3]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[X_1]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           quake.sink %[[MZ_0]] : !quake.wire
// CHECK:           %[[NULL_WIRE_3:.*]] = quake.null_wire
// CHECK:           %[[X_2:.*]]:2 = quake.x {{\[}}%[[H_1]]] %[[NULL_WIRE_3]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X_3:.*]]:2 = quake.x {{\[}}%[[S_1]]] %[[X_2]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[VAL_1:.*]], %[[MZ_1:.*]] = quake.mz %[[X_3]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           quake.sink %[[MZ_1]] : !quake.wire
// CHECK:           %[[S_2:.*]] = quake.s<adj> %[[X_2]]#0 : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_4:.*]] = quake.h %[[S_2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X_4:.*]]:2 = quake.x {{\[}}%[[H_4]]] %[[X_3]]#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.250000e-01 : f64
// CHECK:           %[[RZ_0:.*]] = quake.rz (%[[CONSTANT_0]]) %[[X_4]]#1 : (f64, !quake.wire) -> !quake.wire
// CHECK:           %[[X_5:.*]]:2 = quake.x {{\[}}%[[X_4]]#0] %[[RZ_0]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[H_5:.*]] = quake.h %[[X_5]]#0 : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_3:.*]] = quake.s %[[H_5]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_6:.*]] = quake.h %[[S_3]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_4:.*]] = quake.s %[[H_6]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H_7:.*]] = quake.h %[[S_4]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[Z_0:.*]] = quake.z %[[H_7]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[S_5:.*]] = quake.s<adj> %[[Z_0]] : (!quake.wire) -> !quake.wire
// CHECK:           quake.sink %[[S_5]] : !quake.wire
// CHECK:           quake.sink %[[X_5]]#1 : !quake.wire
// CHECK:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
// CHECK:           %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[XORI_0:.*]] = arith.xori %[[DISCRIMINATE_1]], %[[CONSTANT_1]] : i1
