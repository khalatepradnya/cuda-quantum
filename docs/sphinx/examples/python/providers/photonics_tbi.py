# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

from dataclasses import dataclass

cudaq.set_target("photonics")

# Global variables
zero = 0
one = 1
n_modes = 4
input_state = [2, 1, 3, 1]
d = np.cumsum(input_state)
n_loops = 2
loop_lengths = [1, 2]
sum_loop_lengths = np.cumsum(loop_lengths)
n_beamsplitters = n_loops * n_modes - sum_loop_lengths


@dataclass
class TBIParameters:
    bs_angles: list[float]
    ps_angles: list[float]
    input_state: list = input_state
    loop_lengths: list = loop_lengths
    n_samples: int = 1000000


@cudaq.photonics.kernel
def TBI(parameters: TBIParameters):
    bs_angles = parameters.bs_angles
    ps_angles = parameters.ps_angles

    quds = cudaq.photonics.qudits(level=d, count=n_modes)

    for i in range(n_modes):
        for _ in range(input_state[i]):
            cudaq.photonics.plus(quds[i])

    c = 0
    for ll in range(loop_lengths):
        for i in range(n_modes - ll):
            cudaq.photonics.beam_splitter(quds[i], quds[i + ll], bs_angles[c])
            cudaq.photonics.phase_shift(quds[i], ps_angles[c])
            c += 1

    cudaq.photonics.mz(quds)


def LinearSpacedArray(xs, min, max, N):
    h = (max - min) / (N - 1)
    val = min
    for i in range(len(xs)):
        xs[i] = val
        val += h


bs_angles = [None] * n_beamsplitters
ps_angles = [None] * n_beamsplitters

LinearSpacedArray(bs_angles, np.pi / 3, np.pi / 6, n_beamsplitters)
LinearSpacedArray(ps_angles, np.pi / 3, np.pi / 5, n_beamsplitters)

parameters = TBIParameters(bs_angles, ps_angles)

counts = cudaq.photonics.sample(1000000, TBI, parameters)
counts.dump()
