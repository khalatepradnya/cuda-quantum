# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Run P3 physical Builds with `cudaq.run`, with and without noise."""

# %%
# Import CUDA-Q and the Logical authoring APIs. Every program here is
# Clifford, so the Stim simulator runs it, also with Pauli noise.
import cudaq
import cudaq.logical as cql
from cudaq.logical import codes, gadgets

cudaq.set_target("stim")
arch = cql.architecture
actions = arch.physical_actions
instruments = arch.physical_instruments

# %%
# 1. A physical program authored directly at P3. It returns one Boolean, so
# each shot gives a `bool`.
machine = arch.PhysicalMachine(
    "ExampleMachine",
    resource_classes={
        "qubits":
            arch.ResourceClass("qubit",
                               4,
                               native_actions=actions.clifford_set(),
                               native_instruments=(instruments.MZ,))
    },
)


@cql.physical(machine)
def flip():
    q, = cql.ops.load(cql.ops.acquire(machine.qubits, count=1), state="zero")
    return cql.measure(cql.ops.apply(actions.X, q))


flip_build = cql.compile(flip)
print(flip_build.to_quake())  # The Quake that `cudaq.run` executes.
print("flip:", cudaq.run(flip_build, shots_count=4))

# %%
# 2. A protocol compiled to P3 for a device. It returns two computed
# Booleans, so each shot gives a `list[bool]` in return order.
BARE = codes.BareQubit
builder = cql.devices.DeviceBuilder("ExampleBareDevice", source_module=None)
encoded = builder.qec.bind(
    builder.logical.add_compute(capacity=2),
    architecture=cql.devices.QECArchitecture(
        "ExampleBareArchitecture",
        BARE.default_encoding,
        link_roots=(gadgets.prepare_zero(BARE), gadgets.measure_z(BARE)),
    ),
)
builder.physical.bind(encoded,
                      to=builder.physical.add_qubits(
                          2 * BARE.block.size,
                          native_actions=actions.clifford_set(),
                          native_instruments=(instruments.MZ, instruments.MPP)))
builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
device = builder.build()


@cql.protocol
def parity() -> tuple[bool, bool]:
    a = cql.prepare_zero(cql.ops.allocate_patch(BARE, region="compute"))
    b = cql.prepare_zero(cql.ops.allocate_patch(BARE, region="compute"))
    a, first = cql.mpp(cql.types.Z(a[0]))
    b, second = cql.mpp(cql.types.Z(b[0]))
    cql.discard((a, b))
    return cql.xor(first, second), cql.ops.all_false(first, second)


parity_build = cql.compile(cql.compile(parity),
                           pipeline=cql.compiler.pipelines.physical(),
                           device=device)
print("parity, all_false:", cudaq.run(parity_build, shots_count=4))

# %%
# 3. Noise acts on the gates the Build emits. The same Build runs clean, then
# with a bit flip after each `x`, then clean again.
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(0.1))
shots = 1000
for label, model in (("clean", None), ("p=0.1", noise), ("clean", None)):
    results = cudaq.run(flip_build, shots_count=shots, noise_model=model)
    print(f"flip {label}: {sum(results)} of {shots} shots are True")

# %%
# 4. Calls and runtime control flow are not lowered yet. A program whose
# gadgets compile to `phys.call` reports the operation it cannot translate.


@cql.program
def calls_gadgets() -> bool:
    return cql.measure_z(cql.prepare_zero())


calls_build = cql.compile(calls_gadgets,
                          pipeline=cql.compiler.pipelines.qec(),
                          device=device)
calls_build = cql.compile(calls_build,
                          pipeline=cql.compiler.pipelines.physical(),
                          device=device)
try:
    cudaq.run(calls_build)
except RuntimeError as error:
    print("not supported:", str(error).splitlines()[1])
