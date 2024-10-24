/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BraketServerHelper.h"

namespace cudaq {
std::string get_from_config(BackendConfig config, const std::string &key,
                            const auto &missing_functor) {
  const auto iter = config.find(key);
  auto item = iter != config.end() ? iter->second : missing_functor();
  std::transform(item.begin(), item.end(), item.begin(),
                 [](auto c) { return std::tolower(c); });
  return item;
}
void check_machine_allowed(const std::string &machine) {
  if (Machines.find(machine) == Machines.end()) {
    std::string allowed;
    for (const auto &machine : Machines)
      allowed += machine.first + " ";
    std::stringstream stream;
    stream << "machine " << machine
           << " is not of known braket-machine set: " << allowed;
    throw std::runtime_error(stream.str());
  }
}

// Implementation of the getValueOrDefault function
std::string
BraketServerHelper::getValueOrDefault(const BackendConfig &config,
                                      const std::string &key,
                                      const std::string &defaultValue) const {
  return config.find(key) != config.end() ? config.at(key) : defaultValue;
}

// Initialize the Braket server helper with a given backend configuration
void BraketServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing Amazon Braket Backend.");

  // Fetch machine info before checking emulate because we want to be able to
  // emulate specific machines.

  auto machine = getValueOrDefault(config, "machine", SV1);
  check_machine_allowed(machine);
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

std::map<std::string, std::string>
BraketServerHelper::generateRequestHeader() const {
  std::string apiKey, refreshKey, credentials, timeStr;
  std::map<std::string, std::string> headers{
      {"Authorization", ""},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

std::map<std::string, std::string>
BraketServerHelper::generateRequestHeader(std::string authKey) const {
  std::map<std::string, std::string> headers{
      {"Authorization", authKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

// Create a job for the IonQ quantum computer
ServerJobPayload
BraketServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> jobs;
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage job;
    job["name"] = circuitCode.name;
    job["target"] = backendConfig.at("target");

    job["qubits"] = backendConfig.at("qubits");
    job["shots"] = shots;
    job["input"]["format"] = "qasm2";
    job["input"]["data"] = circuitCode.code;
    jobs.push_back(job);
  }
  // Get the headers
  RestHeaders headers = generateRequestHeader();

  cudaq::info(
      "Created job payload for braket, language is OpenQASM 2.0, targeting {}",
      backendConfig.at("target"));

  // return the payload
  std::string baseUrl = "";
  return std::make_tuple(baseUrl + "job", headers, jobs);
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::BraketServerHelper, braket)