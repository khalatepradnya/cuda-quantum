/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_CONVERSION_PHYSTOQUAKE_PASSES_TD
#define QLX_CONVERSION_PHYSTOQUAKE_PASSES_TD

include "mlir/Pass/PassBase.td"

def PhysToQuake : Pass<"phys-to-quake", "mlir::ModuleOp"> {
  let summary = "Lower a P3 physical graph to an executable Quake entry point";
  let description = [{
    Translates the `phys.graph` named by `entry` into one `func.func` with the
    `cudaq-entrypoint` attribute, in value-semantics Quake. The function is
    named `__nvqpp__mlirgen__<entry>`, as the CUDA-Q front-ends name kernels.
    The module then holds only that function, the intrinsics it calls, and
    the `quake.noOptimization` module attribute. Other graphs in the module
    are erased without being translated.

    The function executes the physical realization the graph states:
      - `phys.acquire` and `phys.release` of a carrier become
        `quake.null_wire` and `quake.sink` at the same program point.
      - `phys.prepare` in state `zero`, `0`, or `loaded`, and `phys.reset` to
        `zero`, become `quake.reset`. The states `plus` and `+` add `h`.
      - `phys.apply` becomes the Quake gate that its action's builtin
        `process` names, once per lane.
      - `phys.measure` becomes `quake.mz`. When the instrument's builtin
        `process` is `measure_x`, `h` comes before the measurement, and
        after it when the state survives. A destructive measurement ends
        with `quake.sink`.
      - `phys.measure_product` collects the parity of its Pauli product on a
        fresh ancilla: each operand is changed to Z around its CX into the
        ancilla, and the ancilla is measured and freed. With `invert`, the
        bit is complemented where it is read.
      - `phys.rotate_product` is `exp(-i angle P / 2)`: basis changes and a
        CX ladder around `rz`, or around `s`, `z`, or `sdg` when the angle is
        a multiple of pi/2, so that Clifford rotations run on Stim.
      - `phys.call` is inlined. `phys.call_template` replays the body of the
        call that `template_event` names, on the carriers that
        `state_aliases` maps the body's resources to. Each carrier's wire is
        found through its resource, because the elided form advances
        carriers without new SSA states.
      - `cflow.if` becomes `cc.if`, and `cflow.repeat` and `cflow.while`
        become `cc.loop`. Each carries every live carrier's wire, then its
        classical values. A record crosses a region boundary as its `i1`.
        A `cflow.while` with `max_iterations` is rejected, because Logical
        does not define what happens at the bound.
      - Actions and instruments resolve in the graph's machine first, then
        in the module, as in the Phys verifier.
      - `phys.condition`, `phys.xor`, and `phys.all_false` become `i1` logic.
        A measurement stays a `!cc.measure_handle` until a Boolean is needed.
      - `phys.delay`, `phys.barrier`, `phys.move`, and `event.fence` emit
        nothing and forward their states.
      - `arith` operations are cloned.

    Every graph result must be Boolean: a measurement record, of any schema,
    or an `i1`. One result becomes the function's `i1` result. Two or more become one
    `!cc.sequence<i1>` in `phys.return` order, returned in heap storage as
    both CUDA-Q front-ends return a list. An `index` input becomes an `i64`
    parameter.

    Any other form fails with a diagnostic on its operation. That covers
    retry, post-selection, resource and factory
    flows, code epoch transitions, patch-granularity resources, parameterized
    or non-builtin processes, quantum inputs, and non-Boolean results.
  }];
  let options = [
    Option<"entry", "entry", "std::string", /*default=*/"\"\"",
           "Symbol of the phys.graph to translate">
  ];
  let dependentDialects = [
    "cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // QLX_CONVERSION_PHYSTOQUAKE_PASSES_TD
