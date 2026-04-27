/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QEC_OPS

include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/QEC/QECDialect.td"
include "mlir/IR/OpBase.td"

class QECOp<string mnemonic, list<Trait> traits = []>
    : Op<QECDialect, mnemonic, traits>;

// Note: QEC ops accept both `!cc.measure_handle` and `AnySignlessInteger`
// (plus the matching `!cc.stdvec` forms) so the same op verifies before and
// after the in-place rewrite performed by `lower-cc-measure-handle`. The pass
// retypes SSA results without legalizing ops, so any narrower constraint here
// would force a parallel rewrite pattern in the pass.
//
// QEC ops are intentionally NOT marked `NoMemoryEffect`: detectors and
// observables are pure declarations from the IR's standpoint, but DCE would
// silently delete them in the absence of users, hiding metadata that backends
// (Stim) and analyses (`erase-qec`) consume by walking the module.

def QEC_DetectorOp : QECOp<"detector"> {
  let summary = "Declare a detector over measurement events.";
  let description = [{
    Declares a parity constraint over one or more measurement events
    identified by their `!cc.measure_handle` (or post-lowering integer)
    values. Under noise-free execution the XOR of the referenced measurement
    outcomes is deterministic; the QEC backend uses this to derive the
    detector error model. Operands may be individual handles or `!cc.stdvec`
    collections thereof. Backends without QEC support drop these ops via the
    `erase-qec` pass.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[cc_MeasureHandleType, AnySignlessInteger,
                        StdvecOf<[cc_MeasureHandleType]>,
                        StdvecOf<[AnySignlessInteger]>]>>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` `(` type($measurements) `)`
  }];
}

def QEC_DetectorsVectorizedOp : QECOp<"detectors_vectorized"> {
  let summary = "Declare N detectors by pairing two measurement vectors.";
  let description = [{
    Given two measurement vectors, declares N detectors where detector i is
    the parity of `prev[i]` and `curr[i]`. `!cc.stdvec` carries no static
    size, so size mismatches are caught at runtime by the backend or by the
    `erase-qec` pass when expanding the op for non-QEC targets. This is the
    natural target for syndrome-extraction loops over consecutive QEC rounds.
  }];
  let arguments = (ins
    AnyTypeOf<[StdvecOf<[cc_MeasureHandleType]>,
               StdvecOf<[AnySignlessInteger]>]>:$prev,
    AnyTypeOf<[StdvecOf<[cc_MeasureHandleType]>,
               StdvecOf<[AnySignlessInteger]>]>:$curr
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $prev `,` $curr `)` attr-dict `:` `(` type($prev) `,` type($curr) `)`
  }];
}

def QEC_LogicalObservableOp : QECOp<"logical_observable"> {
  let summary = "Declare a logical observable over measurement events.";
  let description = [{
    Declares a logical observable as the XOR of measurement outcomes.
    Operands may be individual `!cc.measure_handle` (or post-lowering
    integer) values or `!cc.stdvec` collections thereof. The optional
    `observableIndex` attribute identifies the observable when a circuit
    declares more than one logical observable simultaneously.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[cc_MeasureHandleType, AnySignlessInteger,
                        StdvecOf<[cc_MeasureHandleType]>,
                        StdvecOf<[AnySignlessInteger]>]>>:$measurements,
    DefaultValuedOptionalAttr<I64Attr, "0">:$observableIndex
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` (`index` $observableIndex^)? attr-dict
    `:` `(` type($measurements) `)`
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
