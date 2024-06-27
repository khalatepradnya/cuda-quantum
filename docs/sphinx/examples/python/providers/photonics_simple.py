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
    qutrits = qudits(level=3, count=2) #for given kernel, number of levels should be constant
    plus(qutrits[0])
    plus(qutrits[1])
    plus(qutrits[1])
    mz(qutrits)


counts = cudaq.photonics.sample(photonicsKernel)
print(counts)
