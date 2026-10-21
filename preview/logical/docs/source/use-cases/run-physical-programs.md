# Run Physical Programs

`cudaq.run` runs a P3 physical Build on a CUDA-Q target. The Build translates
its physical graph to Quake with `Build.to_quake()`, and `cudaq.run` executes
that Quake as it executes any kernel: once per shot, with the same argument,
result, and error handling.

```{literalinclude} ../../../examples/standalone/06_run_physical_programs.py
:language: python
:start-at: import cudaq
:caption: Run P3 physical programs, with and without noise (examples/standalone/06_run_physical_programs.py).
```

## Inputs

The input is a P3 Build: a `@cql.physical` program, a `PhysicalBuilder`
graph, or a program or protocol compiled with
`cql.compiler.pipelines.physical()` for a device. The Build's root graph is the
entry point. It must return its results explicitly. Nothing is measured or
returned implicitly, and results are not decoded.

## Results

A physical program returns Booleans only:

- One returned value comes back as a `bool` for each shot.
- Two or more values come back as one `list[bool]` for each shot, in return
  order.
- Each value is a measurement record or a Boolean computed from records, such
  as `cql.xor` or `cql.ops.all_false` in a protocol.

Two consequences follow from this rule:

- A program that returns a one-element list gets a `bool`, not a list.
- A program that returns a tuple gets a list.

`to_quake()` rejects any other result, such as an integer, a float, or a
quantum state, with an error that names the operation. A program without
results translates, and `cudaq.run` then rejects it, as it rejects any kernel
without a result. Arguments pass as they do for an ordinary kernel. A Logical
`int` argument takes a Python `int`.

## What runs

`to_quake()` translates each P3 operation in program order:

- `acquire` allocates a qubit, and `release` frees it, at the same program
  points. A destructive measurement also frees its qubit.
- `prepare` and `reset` to `zero` reset the qubit. The `plus` state adds `h`.
- An action runs the Quake gate that its builtin process names: `h`, `s`,
  `sdg`, `t`, `tdg`, `x`, `y`, `z`, `cx`, `cz`, `swap`, or `reset`. The
  action's symbol name does not matter.
- A Z measurement is `mz`. An X measurement is `h` then `mz`, followed by `h`
  when the qubit survives, so that the qubit keeps the measured X eigenstate.
- A product measurement collects its parity on a fresh ancilla qubit. Each
  qubit is changed to the Z basis around a `cx` into the ancilla, and the
  ancilla is measured and freed, so the measured qubits keep their
  post-measurement state. A negated product returns the complemented bit.
- A product rotation by `angle` is `exp(-i angle P / 2)`: basis changes and a
  `cx` ladder around `rz`. A multiple of pi/2, up to floating-point rounding
  and at most 128 full turns, uses `s`, `z`, `sdg`, or no gate in place of
  `rz`. Then Clifford rotations run on `stim`. A rotation by at most 1e-12 rad
  in magnitude uses no gate. Every other angle keeps `rz`, however close it is
  to a multiple of pi/2.
- A gadget call runs its body in place. A repeated call, which P3 states as a
  call template, runs the same body on its own qubits.
- Runtime control flow runs at run time: `cql.ops.cond` branches on a
  measured bit, and `cql.ops.repeat` repeats its body. A qubit allocated
  inside a branch is freed inside it. `cflow.while` loops until its
  condition reads 0.
- `delay`, `barrier`, `move`, and `fence` emit no gates.

`print(build.to_quake())` shows the exact circuit. Each call returns a new
module, and the Build does not change.

## Noise and targets

- A `noise_model` passed to `cudaq.run` applies to the emitted gates, as it
  does for any kernel. The same Build emits the same circuit with or without
  a model. Delays and moves emit no gates, so they add no noise, and a run does
  not model the schedule's timing, idle periods, or transport.
- X measurements, `plus` preparations, and product measurements and
  rotations run as the gates that `to_quake()` shows. A noise model acts on
  those gates, including the `cx` gates of a product measurement.
- `stim` runs Clifford programs, also with Pauli noise, and suits larger
  programs. It rejects `t` gates and every product rotation that keeps
  `rz`. `density-matrix-cpu` and `nvidia` also apply noise models.
  `qpp-cpu` ignores noise models with a warning, as it does for any kernel.

## Limits

`to_quake()` raises an error that names the first operation it cannot
translate. These forms do not translate:

- A `cflow.while` with `max_iterations`. Logical does not define what
  happens when the loop reaches the bound.
- Retry, post-selection, resource and factory flows, and code epoch
  transitions.
- Patch-granularity resources, where one state stands for a whole code patch.
- Actions with parameters or reservations, such as the trapped-ion `rz`,
  `zz`, and `phased_x` actions. Logical states no gate for them. Processes
  that are not builtin are also rejected.
