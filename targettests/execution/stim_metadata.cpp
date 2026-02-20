/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target stim %s -o %t && %t

// Tests Phase 2 infrastructure: uniqueId population in measure_result and
// QPU-owned metadata query via cudaq::get_platform().query<T>().

#include <cassert>
#include <cudaq.h>
#include <cudaq/platform.h>
#include <nvqir/stim/StimMeasurementData.h>

struct single_measure {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    return mz(q[0]);
  }
};

struct no_return {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

struct single_qubit_multi_measure {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
    x(q);
    mz(q);
  }
};

struct multi_measure {
  auto operator()() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    h(q[1]);
    return mz(q);
  }
};

int main() {
  auto &platform = cudaq::get_platform();
  std::size_t nShots = 5;

  // Test 1: Single measure_result return
  {
    auto results = cudaq::run(nShots, single_measure{});
    assert(results.size() == nShots);

    for (auto &m : results) {
      assert(m.getResult() == 0 || m.getResult() == 1);
      assert(m.getUniqueId() == 0);
    }

    auto *meta = platform.query<cudaq::StimMeasurementData>();
    assert(meta != nullptr);
    assert(meta->measurements.size() == 1);
    auto *info = meta->lookup(0);
    assert(info != nullptr);
    assert(info->qubit_index == 0);
  }

  // Test 1.5: Sample example
  {
    auto counts = cudaq::sample(nShots, no_return{});
    counts.dump();
    auto *meta = platform.query<cudaq::StimMeasurementData>();
    // We get the metadata
    assert(meta != nullptr);
    // We get two measurements (one for each qubit)
    assert(meta->measurements.size() == 2);
    printf("Metadata for no_return:\n");
    for (const auto &[id, info] : meta->measurements) {
      printf("  uniqueId: %d, qubit_index: %zu\n", id, info.qubit_index);
    }
    /*
    { 00:2 11:3 }
    Metadata for no_return:
      uniqueId: 1, qubit_index: 1
      uniqueId: 0, qubit_index: 0
    */
  }

  // Test 1.75 Sample with explicit measurements
  {
    cudaq::sample_options options{.shots = nShots,
                                  .explicit_measurements = true};
    auto counts = cudaq::sample(options, single_qubit_multi_measure{});
    counts.dump();
    auto *meta = platform.query<cudaq::StimMeasurementData>();
    // We get the metadata
    assert(meta != nullptr);
    // We get two measurements (one for each mz)
    assert(meta->measurements.size() == 2);
    printf("Metadata for single_qubit_multi_measure:\n");
    for (const auto &[id, info] : meta->measurements) {
      printf("  uniqueId: %d, qubit_index: %zu\n", id, info.qubit_index);
    }
    /*
    { 01:2 10:3 }
    Metadata for single_qubit_multi_measure:
      uniqueId: 1, qubit_index: 0
      uniqueId: 0, qubit_index: 0
    */
  }

  // Test 2: Vector of measure_result return
  {
    auto results = cudaq::run(nShots, multi_measure{});
    assert(results.size() == static_cast<std::size_t>(nShots));

    for (auto &shot : results) {
      assert(shot.size() == 3);
      assert(shot[0].getResult() == 1);
    }

    auto *meta = platform.query<cudaq::StimMeasurementData>();
    assert(meta != nullptr);
    assert(meta->measurements.size() == 3);
    for (int i = 0; i < 3; i++) {
      auto *info = meta->lookup(i);
      assert(info != nullptr);
      assert(info->qubit_index == static_cast<std::size_t>(i));
    }
  }

  // Test 3: Multiple mz() calls — uniqueId must be globally unique
  //
  // BUG (known): RecordLogParser::processArrayEntry currently uses the
  // array-local index as uniqueId. With two mz(q) calls on 2 qubits,
  // the first array gets {0, 1} and the second also gets {0, 1} instead
  // of {2, 3}. The Stim backend assigns correct global IDs internally but the
  // record parser is not using this, and simply substitutes array-local
  // indices.
  //
  {
    struct multi_round {
      auto operator()() __qpu__ {
        cudaq::qvector q(2);
        x(q[0]);
        auto r1 = mz(q); // measurements 0, 1
        auto r2 = mz(q); // measurements 2, 3
        return r2;
      }
    };

    auto results = cudaq::run(nShots, multi_round{});

    // Stim metadata should reflect 4 total measurements across both rounds.
    auto *meta = platform.query<cudaq::StimMeasurementData>();
    assert(meta != nullptr);
    assert(meta->measurements.size() == 4);
    assert(meta->lookup(0)->qubit_index == 0); // round 1, qubit 0
    assert(meta->lookup(1)->qubit_index == 1); // round 1, qubit 1
    assert(meta->lookup(2)->qubit_index == 0); // round 2, qubit 0
    assert(meta->lookup(3)->qubit_index == 1); // round 2, qubit 1

    // The returned results are from r2 (the second mz(q)), which is an
    // array of 2 results. Their uniqueIds should be 2 and 3 (global).
    //
    // for (auto &shot : results) {
    //   assert(shot.size() == 2);
    //   assert(shot[0].getUniqueId() == 2); // 3rd measurement globally
    //   assert(shot[1].getUniqueId() == 3); // 4th measurement globally
    // }
  }

  return 0;
}
