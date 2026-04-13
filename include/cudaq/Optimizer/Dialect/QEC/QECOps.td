/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QEC_OPS

include "cudaq/Optimizer/Dialect/QEC/QECDialect.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.td"
include "mlir/IR/OpBase.td"

class QECOp<string mnemonic, list<Trait> traits = []>
    : Op<QECDialect, mnemonic, traits>;

def QEC_DetectorOp : QECOp<"detector"> {
  let summary = "Declare a detector over measurement results.";
  let description = [{
    Declares a parity constraint over one or more measurement results.
    Under noise-free execution the XOR of the referenced measurements
    is deterministic. Operands may be individual `!quake.measure` values
    or a `!quake.measurements<?>` collection. This operation has no side
    effects and can be erased for backends that do not support it.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[MeasureType, MeasurementsType]>>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` `(` type($measurements) `)`
  }];
}

def QEC_DetectorsVectorizedOp : QECOp<"detectors_vectorized"> {
  let summary = "Declare N detectors by pairing prev and curr measurement vectors.";
  let description = [{
    Given two collections of measurement results, declares N detectors where
    detector i is the parity of prev[i] and curr[i]. Both operands must be
    `!quake.measurements<N>` with matching sizes.
  }];
  let arguments = (ins
    MeasurementsType:$prev,
    MeasurementsType:$curr
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $prev `,` $curr `)` attr-dict `:` `(` type($prev) `,` type($curr) `)`
  }];
}

def QEC_LogicalObservableOp : QECOp<"logical_observable"> {
  let summary = "Declare a logical observable over measurement results.";
  let description = [{
    Declares a logical observable as a binary sum of measurement results.
    Operands may be individual `!quake.measure` values or a
    `!quake.measurements<?>` collection.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[MeasureType, MeasurementsType]>>:$measurements,
    DefaultValuedOptionalAttr<I64Attr, "0">:$observableIndex
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` (`index` $observableIndex^)? attr-dict `:` `(` type($measurements) `)`
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
