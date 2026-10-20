// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --split-input-file --phys-to-quake=entry=entry --verify-diagnostics %s

// Each case is valid P3 with one form that the pass rejects.

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
  }
  // expected-error@+1 {{'phys.action' op has a process that is not valid JSON}}
  phys.action @op {arity = 1 : i64, process = "h"}
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
  }
  // expected-error@+1 {{'phys.action' op has a process that is not a builtin process, which has no exact lowering to Quake}}
  phys.action @op {arity = 1 : i64, process = "{\"kind\":\"unitary\",\"matrix\":[[0,1],[1,0]]}"}
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
  }
  // expected-error@+1 {{'phys.action' op has a parameterized process, which has no lowering to Quake}}
  phys.action @op {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{\"angle\":0.5}}"}
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

// The trapped-ion builtins state no gate, so they fail closed.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@rz], native_instruments = []}
  }
  phys.action @rz {arity = 1 : i64, parameters = ["angle"], process = "{\"kind\":\"builtin\",\"name\":\"rz\",\"parameters\":{}}"}
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.apply' op applies builtin process 'rz' to 1 states, which has no lowering to Quake}}
    %1 = phys.apply @rz(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

// Reservations carry values that the process does not state.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@x], native_instruments = []}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.apply' op carries reservations ["angle=0.3"], which have no lowering to Quake}}
    %1 = phys.apply @x(%0) {event_id = "e1", reservations = ["angle=0.3"]} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

// An unresolved action or instrument verifies in a partially linked module.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.apply' op names action @h, which resolves to no phys.action}}
    %1 = phys.apply @h(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.measure' op names instrument @mz, which resolves to no phys.instrument}}
    %record = phys.measure @mz(%0) {destructive, event_id = "e1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
    phys.return
  }
}

// -----

// The Phys verifier does not check an instrument declared in the machine.
module {
  phys.machine @m {
    phys.instrument @probe {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}", record_schema = "bit"}
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@probe]}
  }
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.measure' op measures with builtin process 'h', which has no lowering to Quake}}
    %record = phys.measure @probe(%0) {destructive, event_id = "e1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
    phys.return
  }
}

// -----

// One state of a patch-granularity resource stands for a whole code patch.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, footprint_evidence = "one rotated surface-code patch", granularity = "patch", kind = "surface_code_patch", native_actions = [], native_instruments = [], physical_unit_kind = "qubit", physical_units = 17 : i64}
  }
  phys.resource @p {architecture = @m, index = 0 : i64, kind = "surface_code_patch", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    // expected-error@+1 {{'phys.acquire' op acquires resource @p of patch granularity, which phys-to-quake does not simulate}}
    %0 = phys.acquire [@p] {event_id = "e0"} : !phys.state<@p>
    phys.release %0 {event_id = "e1"} : !phys.state<@p>
    phys.return
  }
}

// -----

// The verifier accepts any prepared state name.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.prepare' op prepares state 'minus', which has no lowering to Quake}}
    %1 = phys.prepare %0 {event_id = "e1", state = "minus"} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

// A caller passes only scalars.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
  // expected-error@+1 {{'phys.graph' op takes input 0 of type '!phys.state<@a>', which a kernel cannot receive from its caller}}
  phys.graph @entry on @m : (!phys.state<@a>) -> () {
  ^bb0(%arg0: !phys.state<@a>):
    phys.release %arg0 {event_id = "e0"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.graph @entry on @m : (i1, index) -> (i1, index) {
  ^bb0(%arg0: i1, %arg1: index):
    // expected-error@+1 {{'phys.return' op returns result 1 of type 'index', but a physical kernel returns only Booleans: measurement records and i1 values}}
    phys.return %arg0, %arg1 : i1, index
  }
}

// -----

// A selection sidecar post-selects the graph's shots. The graph and sidecar
// follow test/Model/p3-qec-sidecars.mlir.
module attributes {qlx.profiles = ["p3"]} {
  fabric.code @source_code {distance = 1 : i64, n = 1 : i64, k = 1 : i64, r = 0 : i64, partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}}
  fabric.gadget @source(%patch: !fabric.patch<@source_code>) -> (!fabric.patch<@source_code>, i1) {
    %next, %record = fabric.measure_product %patch {logical_indices = array<i64: 0>, patch_indices = array<i64: 0>, pauli_product = "Z", record = "final"} : (!fabric.patch<@source_code>) -> (!fabric.patch<@source_code>, i1)
    fabric.return %next, %record : !fabric.patch<@source_code>, i1
  }
  fabric.gadget_profile @analysis for @source {
    fabric.success {records = ["source.final.outcome"]}
  }
  phys.machine @arch {
    phys.resource_class @qubits {count = 1 : i64, kind = "qubit", native_actions = ["measure_z"]}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @arch : () -> !phys.state<@q0> attributes {source_protocol = @source} {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1:2 = phys.measure @measure_z(%0) {event_id = "m0", record_id = "readout.final.data0"} : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %1#0 : !phys.state<@q0>
  }
  phys.record_projection @projection for @entry from @source {
    entries = [{instance = "source", physical_record = "readout.final.data0", source_record = "source.final.outcome"}]
  }
  // expected-error@+1 {{'phys.selection_sidecar' op post-selects shots of the entry graph, which phys-to-quake does not support}}
  phys.selection_sidecar @success for @entry {constant = false, expected = false, input_syndromes = [], projection_indices = array<i64: 0>, record_projection = @projection, records = ["readout.final.data0"], source_kind = "profile", source_instance = "source", source_profile = @analysis, source_records = ["source.final.outcome"], source_row = 0 : i64}
}

// -----

// In-graph post-selection, as the native projector emits it.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.graph @entry on @m : (i1) -> () {
  ^bb0(%arg0: i1):
    // expected-error@+1 {{'event.selection' op post-selects shots, which phys-to-quake does not support}}
    event.selection %arg0 {accept_when = false, mode = "abort_on"} : i1
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.graph @entry on @m : () -> () {
    // expected-error@+1 {{'phys.produce_resource' op is part of a resource or factory flow, which phys-to-quake does not support}}
    %0 = phys.produce_resource @t_state at @factory {protocol = #fabric.spec_only<"distill-15to1">, event_id = "e0"} : !phys.resource_payload<@t_state>
    phys.discard_resource_payload %0 {event_id = "e1"} : !phys.resource_payload<@t_state>
    phys.return
  }
}

// -----

// The transition follows test/Model/p3-epoch-transition.mlir.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = []}
  }
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.epoch_transition' op changes the code epoch, and phys-to-quake cannot check that the operations before it realize the declared logical action}}
    %1 = phys.epoch_transition @even to @odd(%0) {event_id = "e1", evidence = "verified", logical_map = {q0 = "q0"}} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e2"} : !phys.state<@a>
    phys.return
  }
}

// -----

// Calls are not lowered yet. A call template replays an earlier call, so a
// program with templates stops at that call. The call follows
// test/Model/p3-structured-control.mlir.
module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {encodings = []}
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = []}
  }
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'phys.call' op has no lowering to Quake}}
    %1 = "phys.call"(%0) <{callee = @attempt, event_id = "call0", instance = "attempt.0", profile = @attempt_profile}> ({
    ^bb0(%current: !phys.state<@a>):
      phys.yield %current : !phys.state<@a>
    }) : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e1"} : !phys.state<@a>
    phys.return
  }
}

// -----

// Product measurements and rotations are not lowered yet. They follow
// test/Model/p3-measure-product.mlir.
module {
  phys.instrument @mpp {kind = "measure_product", variadic, record_schema = "bit", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_product\",\"parameters\":{}}"}
  phys.machine @m {
    phys.resource_class @q {count = 2 : i64, kind = "qubit", native_actions = [], native_instruments = [@mpp]}
  }
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@a, @b] {event_id = "e0"} : !phys.state<@a>, !phys.state<@b>
    // expected-error@+1 {{'phys.measure_product' op has no lowering to Quake}}
    %1:3 = phys.measure_product %0#0, %0#1 {event_id = "e1", instrument = @mpp, paulis = ["X", "Z"], record_id = "r"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>, !phys.record<@bit>)
    phys.release %1#0, %1#1 {event_id = "e2"} : !phys.state<@a>, !phys.state<@b>
    phys.return %1#2 : !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 2 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {capabilities = ["qlx.physical/native_pauli_product_rotation"], index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {capabilities = ["qlx.physical/native_pauli_product_rotation"], index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0:2 = phys.acquire [@a, @b] {event_id = "e0"} : !phys.state<@a>, !phys.state<@b>
    // expected-error@+1 {{'phys.rotate_product' op has no lowering to Quake}}
    %1:2 = phys.rotate_product %0#0, %0#1 {angle = 1.250000e-01 : f64, event_id = "e1", paulis = ["Y", "Z"]} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>)
    phys.release %1#0, %1#1 {event_id = "e2"} : !phys.state<@a>, !phys.state<@b>
    phys.return
  }
}

// -----

// Runtime control flow is not lowered yet. A Python loop or branch in the
// authoring code produces straight-line P3 and is supported.
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@x], native_instruments = []}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : (i1) -> () {
  ^bb0(%arg0: i1):
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'cflow.if' op has no lowering to Quake}}
    %1 = "cflow.if"(%arg0) <{event_id = "if0"}> ({
      %2 = phys.apply @x(%0) {event_id = "x0"} : (!phys.state<@a>) -> !phys.state<@a>
      cflow.yield %2 : !phys.state<@a>
    }, {
      cflow.yield %0 : !phys.state<@a>
    }) : (i1) -> !phys.state<@a>
    phys.release %1 {event_id = "e1"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@x], native_instruments = []}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'cflow.repeat' op has no lowering to Quake}}
    %1 = "cflow.repeat"(%0) <{count = 3 : i64, event_id = "repeat0"}> ({
    ^bb0(%current: !phys.state<@a>):
      %2 = phys.apply @x(%current) {event_id = "x0"} : (!phys.state<@a>) -> !phys.state<@a>
      cflow.yield %2 : !phys.state<@a>
    }) : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "e1"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : (i1) -> () {
  ^bb0(%arg0: i1):
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    // expected-error@+1 {{'cflow.while' op has no lowering to Quake}}
    %1:2 = "cflow.while"(%0, %arg0) <{event_id = "while0", max_iterations = 4 : i64}> ({
    ^bb0(%current: !phys.state<@a>, %predicate: i1):
      "cflow.while_condition"(%predicate, %current, %predicate) : (i1, !phys.state<@a>, i1) -> ()
    }, {
    ^bb0(%current: !phys.state<@a>, %predicate: i1):
      cflow.yield %current, %predicate : !phys.state<@a>, i1
    }) : (!phys.state<@a>, i1) -> (!phys.state<@a>, i1)
    phys.release %1#0 {event_id = "e1"} : !phys.state<@a>
    phys.return
  }
}

// -----

module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.graph @entry on @m : (i1) -> () {
  ^bb0(%arg0: i1):
    // expected-error@+1 {{'cf.assert' op has no lowering to Quake}}
    cf.assert %arg0, "unsupported"
    phys.return
  }
}
