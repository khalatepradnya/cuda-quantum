// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake, cudaq-opt
// RUN: qlx-opt --phys-to-quake=entry=entry %s | %cudaq_opt --pass-pipeline='builtin.module(codegen-for-qir{convert-to=qir})' | FileCheck %s --check-prefix=QIS
// RUN: qlx-opt --phys-to-quake=entry=entry %s | %cudaq_opt --pass-pipeline='builtin.module(aot-prep-pipeline,codegen-for-qir{convert-to=qir-adaptive})' | FileCheck %s --check-prefix=PROFILE

// The final QIS calls keep P3's allocation points and measurement basis.
// @a is allocated at its acquire and released by its destructive
// measurement, and only then is @b allocated. The reset to plus is
// `reset; h`, and the retained X measurement is `h; mz; h`.
//
// The returned list survives QIR profile output. The `.run`
// function that `cudaq.run` executes records the list as an array of three
// Booleans, then each element in return order. Profile output records a
// list only when its length folds to a constant, and drops it without a
// diagnostic otherwise. `quake.mangled_name_map` stands in for the host map
// that `cudaq.run` adds on admission: `aot-prep-pipeline` creates `.run` only
// in a module that has one, and the pass keeps it.
module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__entry = "entry_host"}} {
  phys.machine @machine {
    phys.resource_class @qubits {count = 2 : i64, kind = "qubit", native_actions = [@x], native_instruments = [@mz, @mx]}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.instrument @mx {arity = 1 : i64, kind = "measure_x", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_x\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @b {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @machine : () -> (!phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>) {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.reset %0 {event_id = "e1", state = "plus"} : (!phys.state<@a>) -> !phys.state<@a>
    %2, %kept = phys.measure @mx(%1) {event_id = "e2", record_id = "kept"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %last = phys.measure @mx(%2) {destructive, event_id = "e3", record_id = "last"} : (!phys.state<@a>) -> !phys.record<@bit>
    %3 = phys.acquire [@b] {event_id = "e4"} : !phys.state<@b>
    %4 = phys.apply @x(%3) {event_id = "e5"} : (!phys.state<@b>) -> !phys.state<@b>
    %flip = phys.measure @mz(%4) {destructive, event_id = "e6", record_id = "flip"} : (!phys.state<@b>) -> !phys.record<@bit>
    phys.return %kept, %last, %flip : !phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>
  }
}

// QIS-LABEL: llvm.func @__nvqpp__mlirgen__entry()
// QIS:         %[[A:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__reset(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[A]], %{{.*}}) {registerName = "r00000"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[A]], %{{.*}}) {registerName = "r00001"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[B:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__x(%[[B]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[B]], %{{.*}}) {registerName = "r00002"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[B]]) : (!llvm.ptr) -> ()

// PROFILE-LABEL: llvm.func @__nvqpp__mlirgen__entry.run()
// PROFILE:         %[[LENGTH:.*]] = llvm.mlir.constant(3 : i64) : i64
// PROFILE-NOT:     record_output
// PROFILE:         llvm.mlir.addressof @[[ARRAY:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__array_record_output(%[[LENGTH]], %{{.*}}) : (i64, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[0] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT0:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[1] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT1:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[2] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT2:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE-NOT:     record_output
// PROFILE:         llvm.return
// PROFILE-LABEL: llvm.func @__nvqpp__mlirgen__entry.run.entry()
// PROFILE-DAG:     llvm.mlir.global private constant @[[ARRAY]]("array<i1 x 3>\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT0]]("[0]\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT1]]("[1]\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT2]]("[2]\00")
