/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BraketExecutor.h"
#include "BraketServerHelper.h"

namespace cudaq {
details::future
BraketExecutor::execute(std::vector<KernelExecution> &codesToExecute) {
  auto braketServerHelper = dynamic_cast<BraketServerHelper *>(serverHelper);
  assert(braketServerHelper);
  braketServerHelper->setShots(shots);
  std::string action =
      "{\"braketSchemaHeader\": {\"name\": \"braket.ir.openqasm.program\", "
      "\"version\": \"1\"}, \"source\": \"\", \"inputs\": {}}";

  std::string source = codesToExecute[0].code;

  std::regex include_re("include \".*\";");

  source = std::regex_replace(source, include_re, "");

  source = std::regex_replace(source, std::regex{"\\scx\\s"}, " cnot ");

  auto action_json = nlohmann::json::parse(action);
  action_json["source"] = source;

  // for(char & c : action){
  //   if(c=='\n') c = ' ';
  // }

  // nlohmann::json::string_t s{action};
  action = action_json.dump();
  cudaq::info("OpenQASM 2 action: {}", action);

  Aws::Client::ClientConfiguration clientConfig;

  std::promise<sample_result> p;
  CountsDictionary cd;

  clientConfig.verifySSL = false; // TODO... fix all of this
  Aws::Braket::BraketClient braketClient{clientConfig};
  Aws::STS::STSClient stsClient{clientConfig};
  Aws::S3Crt::ClientConfiguration s3config;
  s3config.verifySSL = false;
  Aws::S3Crt::S3CrtClient s3Client{s3config};

  auto callerIdResponse = stsClient.GetCallerIdentity();

  if (!callerIdResponse.IsSuccess()) {
    p.set_exception(std::make_exception_ptr(std::runtime_error("STS failed")));
    return p.get_future();
  }

  auto account = callerIdResponse.GetResult().GetAccount();

  std::string defaultBucket =
      fmt::format("amazon-braket-{}-{}", clientConfig.region, account);
  std::string defaultPrefix = "tasks-cudaq";

  Aws::Braket::Model::CreateQuantumTaskRequest req;

  auto config = braketServerHelper->getConfig();
  cudaq::info("Backend config: {}, shots {}", config, shots);
  config.insert({"shots", std::to_string(shots)});

  std::string sv1_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1";

  req.SetAction(action);
  req.SetDeviceArn(sv1_arn);
  req.SetShots(shots);
  req.SetOutputS3Bucket(defaultBucket);
  req.SetOutputS3KeyPrefix(defaultPrefix);

  auto response = braketClient.CreateQuantumTask(req);

  if (response.IsSuccess()) {
    std::string taskArn = response.GetResult().GetQuantumTaskArn();
    cudaq::info("Created {}", taskArn);
    Aws::Braket::Model::QuantumTaskStatus taskStatus;
    Aws::Braket::Model::GetQuantumTaskRequest getTaskReq;
    getTaskReq.SetQuantumTaskArn(taskArn);

    auto r = braketClient.GetQuantumTask(getTaskReq);
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
      r = braketClient.GetQuantumTask(getTaskReq);
      taskStatus = r.GetResult().GetStatus();
    } while (taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED &&
             taskStatus != Aws::Braket::Model::QuantumTaskStatus::FAILED &&
             taskStatus != Aws::Braket::Model::QuantumTaskStatus::CANCELLED);

    std::string outBucket = r.GetResult().GetOutputS3Bucket();
    std::string outPrefix = r.GetResult().GetOutputS3Directory();

    cudaq::info("results at {}/{}", outBucket, outPrefix);
    Aws::S3Crt::Model::GetObjectRequest resultRequest;
    resultRequest.SetBucket(outBucket);
    resultRequest.SetKey(fmt::format("{}/results.json", outPrefix));

    auto results = s3Client.GetObject(resultRequest);

    auto resultjson =
        nlohmann::json::parse(results.GetResultWithOwnership().GetBody());

    auto measurements = resultjson.at("measurements");

    for (auto const &m : measurements) {
      std::string bitString = "";
      for (int bit : m) {
        bitString += std::to_string(bit);
      }
      cd[bitString] += 1;
    }

  } else {
    std::cout << "Create error\n" << response.GetError() << "\n";
  }

  ExecutionResult ex_r{cd};

  sample_result result{ex_r};
  p.set_value(result);

  return {p.get_future()};
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::BraketExecutor, braket);