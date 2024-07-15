# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# By default, `qudit` is a `qubit`
globalQuditLevel = 2

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


def processQubitIds(opName, *args):
    """
    Return the qubit unique ID integers for a general tuple of 
    kernel arguments, where all arguments are assumed to be qubit-like 
    (`qvector`, `qview`, `qubit`).
    """
    qubitIds = []
    for a in args:
        if isinstance(a, cudaq_runtime.qubit):
            qubitIds.append(a.id())
        elif isinstance(a, cudaq_runtime.qvector) or isinstance(
                a, cudaq_runtime.qview):
            [qubitIds.append(q.id()) for q in a]
        else:
            raise Exception(
                "invalid argument type passed to {}.__call__".format(opName))
    return qubitIds


def plus(qudit):
    global globalQuditLevel
    op_name = "plusGate"
    qubitId = processQubitIds(op_name, qudit)[0]
    cudaq_runtime.photonics.applyOperation(op_name, [],
                                           [[globalQuditLevel, qubitId]])


def mz(qudit):
    qubitId = processQubitIds("mz", qudit)[0]
    res = cudaq_runtime.measure(qubitId)
    return res

# TODO: Support 'broadcast' operations


class PhotonicsKernelDecorator(object):

    def __init__(self, function, level=2):
        global globalQuditLevel

        self.kernelFunction = function
        globalQuditLevel = level

        self.kernelFunction.__globals__['plus'] = plus
        self.kernelFunction.__globals__['mz'] = mz
        ## TODO: Add remaining gates

    def __call__(self, *args):
        return self.kernelFunction(*args)


def photonics_kernel(function=None, **kwargs):

    def wrapper(function):
        return PhotonicsKernelDecorator(function, **kwargs)

    return wrapper
