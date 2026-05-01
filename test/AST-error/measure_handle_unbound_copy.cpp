/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

// The unbound-handle check at the bool-coercion site must trace through
// copy/move construction. A `measure_handle h2 = h;` (or `h2 = h;`)
// stores the unbound `h` into `h2`'s alloca, which would otherwise look
// "bound" because the alloca has a store. The check must trace the
// stored value back to find the originating alloca.

#include <cudaq.h>

// expected-note@* 0+ {{}}

struct DirectUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h;
    (void)b;
  }
};

struct CopyConstructedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    cudaq::measure_handle h2 = h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h2;
    (void)b;
  }
};

struct CopyAssignedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    cudaq::measure_handle h2;
    h2 = h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h2;
    (void)b;
  }
};

// Sanity: a handle that *is* bound by a measurement must continue to
// pass the check, even when its discriminate site reads a copied alloca.
struct CopyConstructedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    cudaq::measure_handle h2 = h;
    bool b = h2;
    (void)b;
  }
};

struct CopyAssignedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    cudaq::measure_handle h2;
    h2 = h;
    bool b = h2;
    (void)b;
  }
};

// Chained assignment `h3 = h2 = h1;` exercises the value-stack discipline
// of the `operator=` interception: the inner `=` must drop its callee
// value before pushing its result so the outer `=` reads the inner
// result (not the leaked callee) as its RHS.

struct ChainedAssignedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h1;
    cudaq::measure_handle h2;
    cudaq::measure_handle h3;
    h3 = h2 = h1;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h3;
    (void)b;
  }
};

struct ChainedAssignedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h2;
    cudaq::measure_handle h3;
    h3 = h2 = mz(q);
    bool b = h3;
    (void)b;
  }
};
