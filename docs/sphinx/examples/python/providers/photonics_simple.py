# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

cudaq.set_target("photonics")


@cudaq.photonics.kernel
def photonicsKernel():
    qutrits = cudaq.photonics.qudits(level=3, count=2)
    cudaq.photonics.plus(qutrits[0])
    cudaq.photonics.plus(qutrits[1])
    cudaq.photonics.plus(qutrits[1])
    cudaq.photonics.mz(qutrits)


counts = cudaq.photonics.sample(photonicsKernel)
print(counts)
