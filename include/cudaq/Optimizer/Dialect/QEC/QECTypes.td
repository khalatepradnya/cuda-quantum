/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The QEC dialect defines no custom types. Detectors and observables operate
// on `!cc.measure_handle` (the IR alias for `cudaq::measure_handle`) before
// the `lower-cc-measure-handle` pass and on `iN` after it, plus the
// corresponding `!cc.stdvec` collections.

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_TYPES
#define CUDAQ_OPTIMIZER_DIALECT_QEC_TYPES

include "cudaq/Optimizer/Dialect/QEC/QECDialect.td"
include "mlir/IR/OpBase.td"

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_TYPES
