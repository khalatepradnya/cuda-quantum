# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

cudaq.set_target("photonics")


# for given kernel, `qudit` level is constant
@cudaq.photonics_kernel(level=3)
def photonicsKernel():
    qutrits = cudaq.qvector(2)
    plus(qutrits[0])
    plus(qutrits[1])
    plus(qutrits[1])
    mz(qutrits)


counts = cudaq.sample(photonicsKernel)
print(counts)
