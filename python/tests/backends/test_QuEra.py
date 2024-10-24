# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

import json
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
    cudaq.set_target('quera')
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


def test_JSON_payload():
    input = {
        "setup": {
            "ahs_register": {
                "sites": [["0.0", "0.0"], ["0.0", "0.000003"],
                          ["0.0", "0.000006"], ["0.000003", "0.0"],
                          ["0.000003", "0.000003"], ["0.000003", "0.000003"],
                          ["0.000003", "0.000006"]],
                "filling": [1, 1, 1, 1, 1, 0, 0]
            }
        },
        "hamiltonian": {
            "drivingFields": [{
                "amplitude": {
                    "time_series": {
                        "values": ["0.0", "25132700.0", "25132700.0", "0.0"],
                        "times": ["0.0", "3E-7", "0.0000027", "0.000003"]
                    },
                    "pattern": "uniform"
                },
                "phase": {
                    "time_series": {
                        "values": ["0", "0"],
                        "times": ["0.0", "0.000003"]
                    },
                    "pattern": "uniform"
                },
                "detuning": {
                    "time_series": {
                        "values": [
                            "-125664000.0", "-125664000.0", "125664000.0",
                            "125664000.0"
                        ],
                        "times": ["0.0", "3E-7", "0.0000027", "0.000003"]
                    },
                    "pattern": "uniform"
                }
            }],
            "localDetuning": [{
                "magnitude": {
                    "time_series": {
                        "values": ["-125664000.0", "125664000.0"],
                        "times": ["0.0", "0.000003"]
                    },
                    "pattern": ["0.5", "1.0", "0.5", "0.5", "0.5", "0.5"]
                }
            }]
        }
    }
    json_in = json.dumps(input)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
