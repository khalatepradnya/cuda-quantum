/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>

// Here we build up a CUDA-Q kernel with N layers and each
// layer containing an arrangement of random SO(4) rotations. The algorithm
// leverages the CUDA-Q VQE support to compute the ground state of the
// Hydrogen atom.

// The SO4 random entangler written as a CUDA-Q kernel free function
// since this is a pure-device quantum kernel
__qpu__ void so4(cudaq::qubit &q, cudaq::qubit &r,
                 const std::vector<double> &thetas) {
  ry(thetas[0], q);
  ry(thetas[1], r);

  h(r);
  cx(q, r);
  h(r);

  ry(thetas[2], q);
  ry(thetas[3], r);

  h(r);
  cx(q, r);
  h(r);

  ry(thetas[4], q);
  ry(thetas[5], r);

  h(r);
  cx(q, r);
  h(r);
}

// The SO4 fabric CUDA-Q kernel. Keeps track of simple
// arithmetic class members controlling the number of qubits and
// entangling layers.
struct so4_fabric {
  void operator()(std::vector<double> params, int n_qubits,
                  int n_layers) __qpu__ {
    cudaq::qvector q(n_qubits);

    x(q[0]);
    x(q[2]);

    const int block_size = 2;
    int counter = 0;
    for (int i = 0; i < n_layers; i++) {
      // first layer of so4 blocks (even)
      for (int k = 0; k < n_qubits; k += 2) {
        auto subq = q.slice(k, block_size);
        auto so4_params = cudaq::slice_vector(params, counter, 6);
        so4(subq[0], subq[1], so4_params);
        counter += 6;
      }

      // second layer of so4 blocks (odd)
      for (int k = 1; k + block_size < n_qubits; k += 2) {
        auto subq = q.slice(k, block_size);
        auto so4_params = cudaq::slice_vector(params, counter, 6);
        so4(subq[0], subq[1], so4_params);
        counter += 6;
      }
    }
  }
};

int main() {
  // Read in the spin op from file
  std::vector<double> h2_data{0, 0, 0, 0, -0.10647701149499994, 0.0,
                              1, 1, 1, 1, 0.0454063328691,      0.0,
                              1, 1, 3, 3, 0.0454063328691,      0.0,
                              3, 3, 1, 1, 0.0454063328691,      0.0,
                              3, 3, 3, 3, 0.0454063328691,      0.0,
                              2, 0, 0, 0, 0.170280101353,       0.0,
                              2, 2, 0, 0, 0.120200490713,       0.0,
                              2, 0, 2, 0, 0.168335986252,       0.0,
                              2, 0, 0, 2, 0.165606823582,       0.0,
                              0, 2, 0, 0, -0.22004130022499996, 0.0,
                              0, 2, 2, 0, 0.165606823582,       0.0,
                              0, 2, 0, 2, 0.174072892497,       0.0,
                              0, 0, 2, 0, 0.17028010135300004,  0.0,
                              0, 0, 2, 2, 0.120200490713,       0.0,
                              0, 0, 0, 2, -0.22004130022499999, 0.0,
                              15};
  cudaq::spin_op H(h2_data, /*nQubits*/ 4);

  // For 8 qubits, 36 parameters per layer
  int n_layers = 2, n_qubits = H.num_qubits(), block_size = 2, p_counter = 0;
  int n_blocks_per_layer = 2 * (n_qubits / block_size) - 1;
  int n_params = n_layers * 6 * n_blocks_per_layer;
  printf("%d qubit Hamiltonian -> %d parameters\n", n_qubits, n_params);

  // Define the initial parameters and ansatz.
  auto init_params =
      cudaq::random_vector(-1, 1, n_params, std::mt19937::default_seed);

  so4_fabric ansatz;

  // Run VQE with lbfgs + central_difference
  {
    cudaq::optimizers::lbfgs optimizer;
    optimizer.initial_parameters = init_params;
    optimizer.max_eval = 20;
    optimizer.max_line_search_trials = 10;
    cudaq::gradients::central_difference gradient;
    auto [opt_val, opt_params] = cudaq::vqe(ansatz, gradient, H, optimizer,
                                            n_params, n_qubits, n_layers);
    printf("Optimal value = %.16lf\n", opt_val);
    REMOTE_TEST_ASSERT(std::abs(opt_val - -1.1164613629294273) < 1e-3);
  }
  // Run VQE with cobyla
  {
    cudaq::optimizers::cobyla optimizer;
    optimizer.initial_parameters = init_params;
    optimizer.max_eval = 100;
    auto [opt_val, opt_params] =
        cudaq::vqe(ansatz, H, optimizer, n_params, n_qubits, n_layers);
    printf("Optimal value = %.16lf\n", opt_val);
    REMOTE_TEST_ASSERT(std::abs(opt_val - -1.0769400650758392) < 1e-3);
  }
  // Run VQE with cobyla with fixed number of shots
  {
    cudaq::optimizers::cobyla optimizer;
    optimizer.initial_parameters = init_params;
    optimizer.max_eval = 100;
    cudaq::set_random_seed(13);
    auto [opt_val, opt_params] = cudaq::vqe(
        /*shots=*/1000, ansatz, H, optimizer, n_params, n_qubits, n_layers);
    printf("Optimal value = %.16lf\n", opt_val);

    // Increase the error tolerance for the shots-based test because this test
    // needs to sometimes run against a server without the remote VQE
    // capability, so the handling of RNG seeds for back-and-forth iterations of
    // observe's behave slightly differently than a fully remote VQE.
    REMOTE_TEST_ASSERT(std::abs(opt_val - -1.0987425678504421) < 0.015);
  }
  return 0;
}
