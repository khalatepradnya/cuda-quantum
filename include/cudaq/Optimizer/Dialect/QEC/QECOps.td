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
include "mlir/IR/OpBase.td"

class QECOp<string mnemonic, list<Trait> traits = []>
    : Op<QECDialect, mnemonic, traits>;

// TODO: constrain operand types from AnyType to !quake.measure /
// cc.stdvec<!quake.measure> once the prototype stabilizes. AnyType is
// acceptable during prototyping but weakens IR verification. See #gap-9.

def QEC_DetectorOp : QECOp<"detector"> {
  let summary = "Declare a detector over measurement results.";
  let description = [{
    Declares a parity constraint over one or more measurement results.
    Under noise-free execution the XOR of the referenced measurements
    is deterministic. Operands may be individual `!quake.measure` values
    or a `cc.stdvec<!quake.measure>`. This operation has no side effects
    and can be erased for backends that do not support it.
  }];
  let arguments = (ins
    Variadic<AnyType>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` `(` type($measurements) `)`
  }];
}

def QEC_DetectorsVectorizedOp : QECOp<"detectors_vectorized"> {
  let summary = "Declare N detectors by pairing prev and curr measurement vectors.";
  let description = [{
    Given two vectors of measurement results, declares N detectors where
    detector i is the parity of prev[i] and curr[i].
  }];
  let arguments = (ins
    AnyType:$prev,
    AnyType:$curr
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
    `cc.stdvec<!quake.measure>`.
  }];
  let arguments = (ins
    Variadic<AnyType>:$measurements,
    DefaultValuedOptionalAttr<I64Attr, "0">:$observableIndex
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` (`index` $observableIndex^)? attr-dict `:` `(` type($measurements) `)`
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
