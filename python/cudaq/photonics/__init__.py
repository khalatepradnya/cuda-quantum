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


def phase_shift(qudit, phi):
    global globalQuditLevel
    op_name = "phaseShiftGate"
    qubitId = processQubitIds(op_name, qudit)[0]
    cudaq_runtime.photonics.applyOperation(op_name, [phi],
                                           [[globalQuditLevel, qubitId]])


def beam_splitter(q, r, theta):
    global globalQuditLevel
    op_name = "beamSplitterGate"
    qId = processQubitIds(op_name, q)[0]
    rId = processQubitIds(op_name, r)[0]
    cudaq_runtime.photonics.applyOperation(
        op_name, [theta], [[globalQuditLevel, qId], [globalQuditLevel, rId]])


# TODO: Check 'broadcast'
def mz(qudits):
    qubitIds = processQubitIds("mz", qudits)
    if len(qubitIds) == 1:
        return cudaq_runtime.measure(qubitIds[0])
    return [cudaq_runtime.measure(qubitIds[i]) for i in qubitIds]


class PhotonicsKernelDecorator(object):

    def __init__(self, function, level=2):
        global globalQuditLevel

        self.kernelFunction = function
        globalQuditLevel = level

        self.kernelFunction.__globals__['plus'] = plus
        self.kernelFunction.__globals__['phase_shift'] = phase_shift
        self.kernelFunction.__globals__['beam_splitter'] = beam_splitter
        self.kernelFunction.__globals__['mz'] = mz

    def __call__(self, *args):
        return self.kernelFunction(*args)


def photonics_kernel(function=None, **kwargs):

    def wrapper(function):
        return PhotonicsKernelDecorator(function, **kwargs)

    return wrapper
