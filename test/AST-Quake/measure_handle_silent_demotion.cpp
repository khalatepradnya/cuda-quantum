/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// A free `__qpu__` function whose signature carries `cudaq::measure_handle`
// is silently demoted to a device-only helper (the host has no way to
// synthesize a handle, so it cannot be called from host code). The bridge
// emits an aborting host stub at the user's mangled name so a stray host
// call surfaces with a clear runtime diagnostic instead of an unresolved
// symbol or a silent miscompile. The device-side function carries
// `cudaq-kernel` only (no `cudaq-entrypoint`), so
// `cudaq::opt::marshal::lookupHostEntryPointFunc` skips it and later
// passes leave the abort body in place.

#include <cudaq.h>

__qpu__ bool consume_handle(cudaq::measure_handle h) { return h; }

// The device-side kernel keeps the `cudaq-kernel` attribute so it is
// visible to the device pipeline, but does NOT acquire `cudaq-entrypoint`
// (silent demotion).

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_consume_handle.
// CHECK-NOT:       cudaq-entrypoint
// CHECK:           cudaq-kernel
// CHECK:           return

// The host-side stub at the user's mangled name calls the runtime abort
// helper and returns undef. The verifier is satisfied; any host
// invocation aborts at runtime.

// CHECK-LABEL:   func.func @_Z14consume_handleN5cudaq14measure_handleE(
// CHECK-NOT:       cudaq-entrypoint
// CHECK:           call @__nvqpp_measureHandleHostBoundaryAbort() : () -> ()
// CHECK-NEXT:      cc.undef i1
// CHECK-NEXT:      return

// CHECK-LABEL:   func.func private @__nvqpp_measureHandleHostBoundaryAbort()
