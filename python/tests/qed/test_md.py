# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq
import numpy as np

@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()
    

def test_single_qubit():
    @cudaq.kernel
    def kernel() -> int:
        q = cudaq.qubit()
        h(q)
        if cudaq.qed.md(q):
            return -1
        return mz(q)
    
    results = cudaq.run(kernel, shots_count=10)
    print (results)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
