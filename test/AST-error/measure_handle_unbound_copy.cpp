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

// Array of handles: discrimination of an uninitialized element must
// fail; discrimination of any element after at least one element has
// been bound is conservatively accepted (the check is element-coarse,
// see `isBoundHandle` in `ConvertExpr.cpp`).

struct ArrayElementUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle hs[2];
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = hs[1];
    (void)b;
  }
};

struct ArrayElementBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle hs[2];
    hs[0] = mz(q);
    bool b = hs[0];
    (void)b;
  }
};

// Aggregate member of measure_handle: same rule. The discrimination of
// an uninitialized member must fail; once any member has been bound,
// the check accepts subsequent discrimination (coarse-grained).

struct Holder {
  cudaq::measure_handle h;
};

struct AggregateMemberUnbound {
  void operator()() __qpu__ {
    Holder holder;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = holder.h;
    (void)b;
  }
};

struct AggregateMemberBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    Holder holder;
    holder.h = mz(q);
    bool b = holder.h;
    (void)b;
  }
};
