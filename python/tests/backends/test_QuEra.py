# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np
import os
import pytest
from multiprocessing import Process
from network_utils import check_server_connection

try:
    from utils.mock_qpu.quera import startServer
except:
    print("Mock qpu not available, skipping QuEra tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62444


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    os.environ["QUERA_API_KEY"] = "00000000000000000000000000000000"
    # Set the targeted QPU
    cudaq.set_target('quera', machine='aquila')
    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)
    yield "Running the tests."
    # Kill the server, remove the file
    p.terminate()


def test_ahs_hello():
    '''
    Test based on
    https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html
    '''
    a = 5.7e-6
    register = []
    register.append(tuple(np.array([0.5, 0.5 + 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([-0.5, 0.5 + 1 / np.sqrt(2)]) * a))

    time_max = 4e-6  # seconds
    time_ramp = 1e-7  # seconds
    omega_max = 6300000.0  # rad / sec
    delta_start = -5 * omega_max
    delta_end = 5 * omega_max

    # omega = ScalarOperator(lambda t: omega_max
    #                        if time_ramp < t < time_max else 0.0)
    # delta = ScalarOperator(lambda t: delta_end
    #                        if time_ramp < t < time_max else delta_start)
    # phi = ScalarOperator(0.0)

    # # Schedule of time steps.
    # steps = [0.0, time_ramp, time_max - time_ramp, time_max]
    # schedule = Schedule(steps, ["t"])

    # evolution_result = evolve(RydbergHamiltonian(atom_sites=register,
    #                                              amplitude=omega,
    #                                              phase=phi,
    #                                              delta_global=delta),
    #                           schedule=schedule)
    # evolution_result.dump()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
