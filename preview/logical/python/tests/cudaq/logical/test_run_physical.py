# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""`cudaq.run` on P3 physical Builds, through `Build.to_quake()`."""

import gc

import pytest

import cudaq
import cudaq.logical as cql
from cudaq.logical import codes, gadgets
from cudaq.logical.compiler import Build, PhysicalBuilder

requires_gpu = pytest.mark.skipif(cudaq.num_available_gpus() == 0,
                                  reason="the nvidia target needs a GPU")

TARGETS = [
    pytest.param(("qpp-cpu", {}), id="qpp-cpu"),
    pytest.param(("density-matrix-cpu", {}), id="density-matrix-cpu"),
    pytest.param(("stim", {}), id="stim"),
    pytest.param(("nvidia", {}), id="nvidia-fp32", marks=requires_gpu),
    pytest.param(("nvidia", dict(option="fp64")),
                 id="nvidia-fp64",
                 marks=requires_gpu),
]


@pytest.fixture(autouse=True, params=TARGETS)
def target(request):
    name, options = request.param
    cudaq.set_target(name, **options)
    try:
        yield request.param
    finally:
        cudaq.reset_target()


SHOTS = 10

arch = cql.architecture
actions = arch.physical_actions
instruments = arch.physical_instruments
machine = arch.PhysicalMachine(
    "run_physical_machine",
    resource_classes={
        "qubits":
            arch.ResourceClass("qubit",
                               16,
                               native_actions=actions.clifford_set(),
                               native_instruments=(instruments.MZ,
                                                   instruments.MX))
    })


def fresh(count=1, state="zero"):
    return cql.ops.load(cql.ops.acquire(machine.qubits, count=count),
                        state=state)


def flip(state):
    return cql.ops.apply(actions.X, state)


@cql.physical(machine)
def x_then_mz():
    # No preparation: a carrier left in |1> by an earlier shot would give
    # False.
    q, = cql.ops.acquire(machine.qubits, count=1)
    return cql.measure(flip(q))


@cql.physical(machine)
def bell():
    a, b = fresh(2)
    a, b = cql.ops.apply(actions.CX, cql.ops.apply(actions.H, a), b)
    return [cql.measure(a), cql.measure(b)]


@cql.physical(machine)
def saved_values():
    # A saved bit survives a reset and a reacquired carrier. The measurement
    # of `b` is not returned, and the result order is the return order.
    a, b = fresh(2)
    a, saved = cql.measure(flip(a), destructive=False)
    cql.measure(b)
    later = cql.measure(cql.reset(a))
    reused, = fresh()
    return later, saved, cql.measure(flip(reused))


@cql.physical(machine)
def syndrome():
    # Four parity checks of a 5-bit chain with an error on bit 1.
    data, checks = list(fresh(5)), list(fresh(4))
    data[1] = flip(data[1])
    for i in range(4):
        data[i], checks[i] = cql.ops.apply(actions.CX, data[i], checks[i])
        data[i + 1], checks[i] = cql.ops.apply(actions.CX, data[i + 1],
                                               checks[i])
    bits = []
    for i in range(4):
        checks[i], bit = cql.measure(checks[i], destructive=False)
        bits.append(bit)
    cql.ops.release(tuple(checks) + tuple(data))
    return bits


@cql.physical(machine)
def x_basis():
    # A retained X measurement keeps the X eigenstate: without the basis
    # restore, the second bit would be random, and `h` would not give |0>.
    q, = fresh(state="plus")
    q, first = cql.measure(q, basis=arch.Basis.X, destructive=False)
    q, second = cql.measure(q, basis=arch.Basis.X, destructive=False)
    return first, second, cql.measure(cql.ops.apply(actions.H, q))


def reset_to_plus():
    builder = PhysicalBuilder("reset_to_plus", architecture=machine)
    q, = builder.acquire(machine.qubits, count=1)
    q, = builder.reset((q,), state="plus")
    q, first = builder.measure(q, basis="x", destructive=False)
    return builder.finish(first, builder.measure(q, basis="x"))


@cql.physical(machine)
def one_element_list():
    q, = fresh()
    return [cql.measure(flip(q))]


def alternating_bits(length):
    """One carrier at a time, so the list fits every target."""

    @cql.physical(machine, name=f"alternating_bits_{length}")
    def body():
        bits = []
        for i in range(length):
            q, = fresh()
            bits.append(cql.measure(flip(q) if i % 2 else q))
        return bits

    return lambda: cql.compile(body)


def physical(definition):
    return lambda: cql.compile(definition)


@pytest.mark.parametrize("make, expected", [
    pytest.param(physical(x_then_mz), True, id="x_then_mz"),
    pytest.param(physical(saved_values), [False, True, True],
                 id="saved_values"),
    pytest.param(physical(syndrome), [True, True, False, False], id="syndrome"),
    pytest.param(physical(x_basis), [False, False, False], id="x_basis"),
    pytest.param(reset_to_plus, [False, False], id="reset_to_plus"),
    pytest.param(physical(one_element_list), True, id="one_element_list"),
    pytest.param(alternating_bits(19), [i % 2 == 1 for i in range(19)],
                 id="list_19"),
    pytest.param(alternating_bits(40), [i % 2 == 1 for i in range(40)],
                 id="list_40"),
])
def test_results(make, expected):
    results = cudaq.run(make(), shots_count=SHOTS)
    assert results == [expected] * SHOTS
    assert types_of(results) == types_of([expected] * SHOTS)


def types_of(results):
    return [[type(bit)
             for bit in result] if isinstance(result, list) else type(result)
            for result in results]


def test_bell_pair():
    results = cudaq.run(cql.compile(bell), shots_count=100)
    assert len(results) == 100
    assert set(map(tuple, results)) == {(False, False), (True, True)}


@cudaq.kernel
def ordinary_bool() -> bool:
    q = cudaq.qubit()
    x(q)
    return mz(q)


@cudaq.kernel
def ordinary_bool_list() -> list[bool]:
    q = cudaq.qvector(3)
    x(q[1])
    x(q[2])
    return [mz(q[0]), mz(q[1]), mz(q[2])]


@pytest.mark.parametrize("ordinary, definition",
                         [(ordinary_bool, x_then_mz),
                          (ordinary_bool_list, saved_values)])
def test_results_match_ordinary_run(ordinary, definition):
    expected = cudaq.run(ordinary, shots_count=2)
    results = cudaq.run(cql.compile(definition), shots_count=2)
    assert results == expected and types_of(results) == types_of(expected)


def test_noise_model_applies_to_emitted_gates(target):
    # A certain bit flip after each `x` undoes it. Of these targets, only
    # density-matrix-cpu takes a channel with probability 1.
    if target[0] != "density-matrix-cpu":
        pytest.skip("needs a channel with probability 1")
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))
    for kernel in (ordinary_bool, cql.compile(x_then_mz)):
        assert cudaq.run(kernel, shots_count=SHOTS,
                         noise_model=noise) == [False] * SHOTS


BARE = codes.BareQubit


def bare_device():
    """One physical qubit per logical qubit, so programs stay small."""
    architecture = cql.devices.QECArchitecture(
        "run_physical_bare_architecture",
        BARE.default_encoding,
        link_roots=(gadgets.prepare_zero(BARE), gadgets.measure_z(BARE)),
    )
    builder = cql.devices.DeviceBuilder("run_physical_bare_device",
                                        source_module=None)
    encoded = builder.qec.bind(builder.logical.add_compute(capacity=2),
                               architecture=architecture)
    builder.physical.bind(encoded,
                          to=builder.physical.add_qubits(
                              2 * BARE.block.size,
                              native_actions=actions.clifford_set(),
                              native_instruments=(instruments.MZ,
                                                  instruments.MPP)))
    builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
    return builder.build()


DEVICE = bare_device()


def physical_build(definition):
    """Compiles a program or protocol to P3 through the native projector."""
    if isinstance(definition, cql.protocols.ProtocolDefinition):
        source = cql.compile(definition)
    else:
        source = cql.compile(definition,
                             pipeline=cql.compiler.pipelines.qec(),
                             device=DEVICE)
    build = cql.compile(source,
                        pipeline=cql.compiler.pipelines.physical(),
                        device=DEVICE)
    # The Python projector, the fallback, emits product measurements.
    assert "fabric-to-phys@0.3" in [
        record.producer
        for record in build.evidence
        if record.kind == "physical_projection_verification"
    ]
    return build


@cql.program
def flag_from_arguments(n: int, theta: float, flag: bool) -> bool:
    return flag


@cudaq.kernel
def ordinary_flag_from_arguments(n: int, theta: float, flag: bool) -> bool:
    return flag


@cql.protocol
def computed_booleans() -> tuple[bool, bool]:
    a = cql.prepare_zero(cql.ops.allocate_patch(BARE, region="compute"))
    b = cql.prepare_zero(cql.ops.allocate_patch(BARE, region="compute"))
    a, first = cql.mpp(cql.types.Z(a[0]))
    b, second = cql.mpp(cql.types.Z(b[0]))
    cql.discard((a, b))
    return cql.xor(first, second), cql.ops.all_false(first, second)


@pytest.mark.parametrize("flag", [False, True])
def test_arguments(flag):
    # Ordinary run is the control for the same signature.
    for kernel in (ordinary_flag_from_arguments,
                   physical_build(flag_from_arguments)):
        assert cudaq.run(kernel, 3, 0.5, flag, shots_count=2) == [flag] * 2


def test_computed_booleans():
    assert cudaq.run(physical_build(computed_booleans),
                     shots_count=SHOTS) == [[False, True]] * SHOTS


def test_run_contract_matches_ordinary_run():
    build = cql.compile(x_then_mz)
    assert cudaq.run(build, shots_count=0) == []
    for kernel in (ordinary_bool, build):
        with pytest.raises(RuntimeError, match="Invalid `shots_count`"):
            cudaq.run(kernel, shots_count=-1)
        with pytest.raises(RuntimeError, match="1 given and 0 expected"):
            cudaq.run(kernel, True)

    # `to_quake()` still shows a program without results.
    @cql.physical(machine)
    def no_result():
        q, = fresh()
        cql.ops.release((flip(q),))

    no_results = cql.compile(no_result)
    assert ("func.func @__nvqpp__mlirgen__no_result() attributes"
            in str(no_results.to_quake()))
    with pytest.raises(RuntimeError,
                       match="a runnable kernel must return a value"):
        cudaq.run(no_results)


@cql.program
def int_result(n: int) -> int:
    return n


@cql.program
def float_result(theta: float) -> float:
    return theta


@cql.program
def mixed_result(n: int, flag: bool) -> tuple[bool, int]:
    return flag, n


@cql.physical(machine)
def state_result():
    q, = fresh()
    return flip(q)


@cql.program
def calls_gadgets() -> bool:
    return cql.measure_z(cql.prepare_zero())


@pytest.mark.parametrize("make, error, message", [
    pytest.param(lambda: physical_build(int_result),
                 RuntimeError,
                 "returns result 0 of type 'index', but a physical kernel "
                 "returns only Booleans",
                 id="int"),
    pytest.param(lambda: physical_build(float_result),
                 RuntimeError,
                 "returns result 0 of type 'f64'",
                 id="float"),
    pytest.param(lambda: physical_build(mixed_result),
                 RuntimeError,
                 "returns result 1 of type 'index'",
                 id="mixed"),
    pytest.param(physical(state_result),
                 RuntimeError,
                 r"returns result 0 of type '!phys.state<@state_result_",
                 id="state"),
    pytest.param(lambda: physical_build(calls_gadgets),
                 RuntimeError,
                 "'phys.call' op has no lowering to Quake",
                 id="call"),
    pytest.param(physical(calls_gadgets),
                 ValueError,
                 "requires a Build of a physical graph, not of a program",
                 id="not_physical"),
])
def test_unsupported_builds(make, error, message):
    with pytest.raises(error, match=message):
        cudaq.run(make())


class QuakeSource:
    """An object whose `to_quake()` returns `make()`, counting the calls."""

    def __init__(self, make):
        self.make = make
        self.conversions = 0

    def to_quake(self):
        self.conversions += 1
        return self.make()


def test_to_quake_inspection_and_lifetime():
    build = cql.compile(x_then_mz)
    source = build.to_mlir()
    module = build.to_quake()
    text = str(module)
    assert "func.func @__nvqpp__mlirgen__x_then_mz() -> i1" in text
    assert "quake.noOptimization" in text
    assert "phys." not in text and "qlx." not in text
    assert build.to_quake() is not module and build.to_mlir() == source

    # The module outlives its Build, and a replayed Build runs without the
    # callable that defined it.
    @cql.physical(machine)
    def local_flip():
        q, = fresh()
        return cql.measure(flip(q))

    payload = cql.compile(local_flip).serialize()
    del build, local_flip
    gc.collect()
    assert cudaq.run(QuakeSource(lambda: module), shots_count=2) == [True, True]
    assert cudaq.run(Build.replay(payload), shots_count=2) == [True, True]


def test_runs_are_independent(target):
    # Two programs whose graphs share one name.
    @cql.physical(machine, name="shared_name")
    def flipped():
        q, = fresh()
        return cql.measure(flip(q))

    @cql.physical(machine, name="shared_name")
    def unflipped():
        q, = fresh()
        return cql.measure(q)

    first = QuakeSource(cql.compile(flipped).to_quake)
    second = cql.compile(unflipped)
    for _ in range(2):
        assert cudaq.run(first, shots_count=SHOTS) == [True] * SHOTS
        assert cudaq.run(second, shots_count=SHOTS) == [False] * SHOTS
    assert first.conversions == 2

    # A failed run, or a run on another target, leaves the next run unaffected.
    with pytest.raises(RuntimeError, match="1 given and 0 expected"):
        cudaq.run(first, True)
    name, options = target
    try:
        cudaq.set_target("stim" if name == "qpp-cpu" else "qpp-cpu")
        assert cudaq.run(first, shots_count=2) == [True, True]
    finally:
        cudaq.set_target(name, **options)
    assert cudaq.run(second, shots_count=2) == [False, False]
