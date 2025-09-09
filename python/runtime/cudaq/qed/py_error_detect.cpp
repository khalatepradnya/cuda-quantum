/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qed/error_detection.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

/// @brief Bind the Quantum Error Detection module.
void bindQED(py::module &mod) {
  mod.def(
      "md", [](qview<> &q) { return qed::md(q); },
      "Detect errors on the provided qubits. Returns a vector of bools where "
      "true indicates an error was detected on the corresponding qubit.",
      py::arg("qubits"));

  mod.def(
      "md", [](qubit &q) { return qed::md(q); },
      "Detect errors on a single qubit. Returns true if an error was detected.",
      py::arg("qubit"));
}

} // namespace cudaq

PYBIND11_MODULE(cudaq_qed, m) {
  m.doc() = "Python bindings for CUDA-Q Quantum Error Detection";
  cudaq::bindQED(m);
}
