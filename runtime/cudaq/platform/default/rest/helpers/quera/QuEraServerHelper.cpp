/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../braket/BraketServerHelper.h"

namespace cudaq {
class QuEraServerHelper : public cudaq::BraketServerHelper {
public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "quera"; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;
};

void QuEraServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing QuEra via Amazon Braket Backend.");

  // Hard-coded for now
  auto machine = Aquila;

  config["machine"] = machine;
  config["target"] = machine;
  cudaq::info("Running on machine {}", machine);

  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    cudaq::info("Emulation is enabled, ignore all braket connection specific "
                "information.");
    backendConfig = std::move(config);
    return;
  }
  config["version"] = "v0.3";
  config["user_agent"] = "cudaq/0.3.0";
  config["qubits"] = Machines.at(machine);
  // Construct the API job path
  config["job_path"] = "/tasks"; // config["url"] + "/tasks";
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  parseConfigForCommonParams(config);

  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Create a job for the QuEra quantum computer
ServerJobPayload
QuEraServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> jobs;
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage job;
    job["name"] = circuitCode.name;
    job["target"] = backendConfig.at("target");

    // job["qubits"] = backendConfig.at("qubits");
    // job["shots"] = shots;
    // job["input"]["format"] = "qasm2";
    // job["input"]["data"] = circuitCode.code;
    jobs.push_back(job);
  }
  // Get the headers
  RestHeaders headers = generateRequestHeader();

  cudaq::info("Created job payload for braket, language is OpenQASM 2.0, "
              "targeting {}",
              backendConfig.at("target"));

  // return the payload
  std::string baseUrl = "";
  return std::make_tuple(baseUrl + "job", headers, jobs);
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuEraServerHelper, quera)
