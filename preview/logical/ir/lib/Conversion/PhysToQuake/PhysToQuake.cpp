/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Lowers the P3 `phys.graph` named by the `entry` option to one Quake entry
// point in value semantics, so that CUDA-Q can execute the physical
// realization. PhysToQuakePasses.td states the translation rules.

#include "qlx/Conversion/PhysToQuakePasses.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include <cmath>

namespace qlx {
#define GEN_PASS_DEF_PHYSTOQUAKE
#include "qlx/Conversion/PhysToQuakePasses.h.inc"
} // namespace qlx

using namespace mlir;
namespace phys = qlx::phys;
namespace quake = cudaq::quake;
namespace cc = cudaq::cc;

/// Returns the builtin process name that an action or instrument declares, or
/// failure, with a diagnostic on `declaration`, for any other process.
static FailureOr<std::string> builtinProcess(Operation *declaration,
                                             StringRef process) {
  auto parsed = llvm::json::parse(process);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return declaration->emitOpError("has a process that is not valid JSON");
  }
  auto *object = parsed->getAsObject();
  auto kind = object ? object->getString("kind") : std::nullopt;
  auto name = object ? object->getString("name") : std::nullopt;
  if (!kind || *kind != "builtin" || !name)
    return declaration->emitOpError(
        "has a process that is not a builtin process, which has no exact "
        "lowering to Quake");
  if (auto *parameters = object->getObject("parameters");
      parameters && !parameters->empty())
    return declaration->emitOpError(
        "has a parameterized process, which has no lowering to Quake");
  return name->str();
}

/// Resolves a symbol the way the Phys verifier resolves an action: in the
/// graph's machine first, then from `user` outward.
static Operation *resolveInMachine(SymbolTableCollection &symbols,
                                   phys::ArchitectureOp machine,
                                   Operation *user, FlatSymbolRefAttr symbol) {
  if (Operation *declared = symbols.lookupSymbolIn(machine, symbol.getAttr()))
    return declared;
  return symbols.lookupNearestSymbolFrom(user, symbol);
}

/// Builds one Quake gate on wires and returns its output wires, controls
/// first.
template <typename GateOp>
static SmallVector<Value> emitGate(OpBuilder &builder, Location loc, bool adj,
                                   ValueRange controls, ValueRange targets,
                                   ValueRange parameters = {}) {
  SmallVector<Type> wireTypes(controls.size() + targets.size(),
                              quake::WireType::get(builder.getContext()));
  auto gate = GateOp::create(builder, loc, wireTypes, adj, parameters, controls,
                             targets, DenseBoolArrayAttr{});
  return SmallVector<Value>(gate.getWires());
}

static Value emitH(OpBuilder &builder, Location loc, Value wire) {
  return emitGate<quake::HOp>(builder, loc, false, {}, wire).front();
}

static Value emitS(OpBuilder &builder, Location loc, bool adj, Value wire) {
  return emitGate<quake::SOp>(builder, loc, adj, {}, wire).front();
}

/// Applies `V` with `V P V^dagger = Z` for the Pauli label `pauli`, or its
/// inverse when `undo` is set, so that a product of Paulis becomes a product
/// of Z.
static Value emitBasisChange(OpBuilder &builder, Location loc, StringRef pauli,
                             bool undo, Value wire) {
  if (pauli == "X")
    return emitH(builder, loc, wire);
  if (pauli != "Y")
    return wire;
  // `H Sdg Y S H = Z`.
  if (undo)
    return emitS(builder, loc, /*adj=*/false, emitH(builder, loc, wire));
  return emitH(builder, loc, emitS(builder, loc, /*adj=*/true, wire));
}

static Value emitReset(OpBuilder &builder, Location loc, Value wire) {
  return quake::ResetOp::create(builder, loc, TypeRange{wire.getType()}, wire)
      ->getResult(0);
}

/// Applies the builtin gate `name` to the wires of one lane, in the action's
/// operand order. Returns the lane's output wires, or failure when the name
/// has no exact Quake gate of that arity.
static FailureOr<SmallVector<Value>> emitBuiltinGate(OpBuilder &builder,
                                                     Location loc,
                                                     StringRef name,
                                                     ValueRange lane) {
  if (lane.size() == 1) {
    if (name == "h")
      return emitGate<quake::HOp>(builder, loc, false, {}, lane);
    if (name == "x")
      return emitGate<quake::XOp>(builder, loc, false, {}, lane);
    if (name == "y")
      return emitGate<quake::YOp>(builder, loc, false, {}, lane);
    if (name == "z")
      return emitGate<quake::ZOp>(builder, loc, false, {}, lane);
    if (name == "s" || name == "sdg")
      return emitGate<quake::SOp>(builder, loc, name == "sdg", {}, lane);
    if (name == "t" || name == "tdg")
      return emitGate<quake::TOp>(builder, loc, name == "tdg", {}, lane);
    if (name == "reset")
      return SmallVector<Value>{emitReset(builder, loc, lane.front())};
  }
  if (lane.size() == 2) {
    if (name == "cx")
      return emitGate<quake::XOp>(builder, loc, false, lane.take_front(),
                                  lane.take_back());
    if (name == "cz")
      return emitGate<quake::ZOp>(builder, loc, false, lane.take_front(),
                                  lane.take_back());
    if (name == "swap")
      return emitGate<quake::SwapOp>(builder, loc, false, {}, lane);
  }
  return failure();
}

namespace {

/// Translates one `phys.graph` into the body of a `func.func`.
class GraphLowering {
public:
  GraphLowering(ModuleOp module, phys::GraphOp graph)
      : module(module), graph(graph), builder(module.getContext()) {}

  /// Builds the entry point, or returns failure after a diagnostic.
  FailureOr<func::FuncOp> lower();

private:
  LogicalResult checkSidecars();
  FailureOr<FunctionType> entryType();
  LogicalResult lowerOp(Operation *op);
  LogicalResult checkCarrier(Operation *op, Value state);
  LogicalResult lowerAcquire(phys::AcquireOp op);
  LogicalResult lowerRelease(phys::ReleaseOp op);
  LogicalResult lowerPreparation(Operation *op, StringRef state,
                                 ValueRange inputs);
  LogicalResult lowerApply(phys::ApplyOp op);
  LogicalResult lowerMeasure(phys::MeasureOp op);
  LogicalResult lowerMeasureProduct(phys::MeasureProductOp op);
  LogicalResult lowerRotateProduct(phys::RotateProductOp op);
  LogicalResult lowerCall(phys::CallOp op);
  LogicalResult lowerCallTemplate(phys::CallTemplateOp op);
  LogicalResult lowerBody(phys::CallOp call, ValueRange inputs,
                          ValueRange outputs);
  LogicalResult lowerIf(qlx::cflow::IfOp op);
  LogicalResult lowerRepeat(qlx::cflow::RepeatOp op);
  LogicalResult lowerWhile(qlx::cflow::WhileOp op);
  LogicalResult lowerOps(Block &block);
  SmallVector<Attribute> liveCarriers();
  void enterRegion(ValueRange arguments, ArrayRef<Attribute> carriers,
                   ValueRange physArguments);
  SmallVector<Value> carriedValues(ArrayRef<Attribute> carriers,
                                   ValueRange physValues);
  void mapResults(ArrayRef<Attribute> carriers, ValueRange physResults,
                  ValueRange results);
  LogicalResult lowerCondition(phys::ConditionOp op);
  LogicalResult lowerAllFalse(phys::AllFalseOp op);
  LogicalResult lowerReturn(phys::ReturnOp op);
  Value toBoolean(Value physValue);
  Attribute carrierOf(Value state);
  Value wireOf(Value state);
  void setWire(Value state, Value wire);

  ModuleOp module;
  phys::GraphOp graph;
  phys::ArchitectureOp machine;
  OpBuilder builder;
  SymbolTableCollection symbols;
  /// Classical values and records. States are tracked in `wires`.
  IRMapping values;
  /// The current wire of each live carrier, keyed by its physical resource.
  /// Note: An elided `phys.call_template` advances its carriers without new
  /// SSA states, and later ops keep using the states from before it. So a
  /// state's wire is found through its resource, never through its SSA value.
  DenseMap<Attribute, Value> wires;
  /// While a template replays a canonical call body, maps each resource that
  /// the body names to the carrier that the invocation acts on.
  DenseMap<Attribute, Attribute> aliases;
  /// The `phys.call` that each `event_id` names, for `phys.call_template`.
  llvm::StringMap<phys::CallOp> canonicalCalls;
  /// Handles of inverted product measurements, whose bit is complemented
  /// where a Boolean is consumed.
  DenseSet<Value> invertedHandles;
};

} // namespace

/// A record or an `i1` is Boolean. Every record that the pass sees comes from
/// a measurement it lowers to one `quake.mz`, whatever the record's schema.
static bool isBoolean(Type type) {
  return isa<phys::RecordType>(type) || type.isInteger(1);
}

LogicalResult GraphLowering::checkSidecars() {
  // A selection sidecar post-selects the graph's shots, which `run` cannot do.
  for (auto sidecar : module.getOps<phys::SelectionSidecarOp>())
    if (sidecar.getGraph() == graph.getSymName())
      return sidecar.emitOpError(
          "post-selects shots of the entry graph, which phys-to-quake does "
          "not support");
  return success();
}

FailureOr<FunctionType> GraphLowering::entryType() {
  FunctionType graphType = graph.getFunctionType();
  SmallVector<Type> inputs;
  for (auto [i, type] : llvm::enumerate(graphType.getInputs())) {
    if (!type.isIntOrIndexOrFloat())
      return graph.emitOpError()
             << "takes input " << i << " of type " << type
             << ", which a kernel cannot receive from its caller";
    // Python passes an `int` as `i64`, and Logical's `int` is `index`.
    inputs.push_back(isa<IndexType>(type) ? builder.getI64Type() : type);
  }
  auto ret = cast<phys::ReturnOp>(graph.getBody().front().getTerminator());
  for (auto [i, type] : llvm::enumerate(graphType.getResults()))
    if (!isBoolean(type))
      return ret.emitOpError()
             << "returns result " << i << " of type " << type
             << ", but a physical kernel returns only Booleans: measurement "
                "records and i1 values";
  SmallVector<Type> results;
  if (graphType.getNumResults() == 1)
    results.push_back(builder.getI1Type());
  else if (graphType.getNumResults() > 1)
    results.push_back(cc::SequenceType::get(builder.getI1Type()));
  return builder.getFunctionType(inputs, results);
}

FailureOr<func::FuncOp> GraphLowering::lower() {
  if (failed(checkSidecars()))
    return failure();
  auto type = entryType();
  if (failed(type))
    return failure();
  // The Phys verifier guarantees that the architecture resolves.
  machine = symbols.lookupNearestSymbolFrom<phys::ArchitectureOp>(
      graph, graph.getArchitectureAttr());
  // The Phys verifier guarantees that each template names exactly one
  // earlier `phys.call` of this graph.
  graph.walk([&](phys::CallOp call) {
    if (auto event = call.getEventIdAttr())
      canonicalCalls[event.getValue()] = call;
  });

  // The entry point has no parent until the pass erases the source ops.
  // `builder` has no insertion point yet. The front-ends' kernel prefix keeps
  // the name apart from the intrinsics the pass loads.
  auto entry = func::FuncOp::create(
      builder, graph.getLoc(),
      (Twine(cudaq::runtime::cudaqGenPrefixName) + graph.getSymName()).str(),
      *type);
  entry->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
  entry->setAttr(cudaq::kernelAttrName, builder.getUnitAttr());
  Block *body = entry.addEntryBlock();
  builder.setInsertionPointToStart(body);
  for (auto [graphArg, entryArg] :
       llvm::zip(graph.getBody().getArguments(), body->getArguments())) {
    Value value = entryArg;
    if (isa<IndexType>(graphArg.getType()))
      value = arith::IndexCastOp::create(builder, graphArg.getLoc(),
                                         builder.getIndexType(), entryArg);
    values.map(graphArg, value);
  }
  for (Operation &op : graph.getBody().front())
    if (failed(lowerOp(&op))) {
      entry.erase();
      return failure();
    }
  return entry;
}

LogicalResult GraphLowering::lowerOp(Operation *op) {
  Location loc = op->getLoc();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](phys::AcquireOp acquire) { return lowerAcquire(acquire); })
      .Case([&](phys::ReleaseOp release) { return lowerRelease(release); })
      .Case([&](phys::PrepareOp prepare) {
        return lowerPreparation(prepare, prepare.getState(),
                                prepare.getInputs());
      })
      .Case([&](phys::ResetOp reset) {
        return lowerPreparation(reset, reset.getState(), reset.getInputs());
      })
      .Case([&](phys::ApplyOp apply) { return lowerApply(apply); })
      .Case([&](phys::MeasureOp measure) { return lowerMeasure(measure); })
      .Case([&](phys::MeasureProductOp measure) {
        return lowerMeasureProduct(measure);
      })
      .Case([&](phys::RotateProductOp rotate) {
        return lowerRotateProduct(rotate);
      })
      .Case([&](phys::CallOp call) { return lowerCall(call); })
      .Case([&](phys::CallTemplateOp invocation) {
        return lowerCallTemplate(invocation);
      })
      .Case([&](qlx::cflow::IfOp ifOp) { return lowerIf(ifOp); })
      .Case([&](qlx::cflow::RepeatOp repeat) { return lowerRepeat(repeat); })
      .Case([&](qlx::cflow::WhileOp loop) { return lowerWhile(loop); })
      .Case([&](phys::ConditionOp condition) {
        return lowerCondition(condition);
      })
      .Case([&](phys::XorOp xorOp) {
        values.map(xorOp.getResult(),
                   arith::XOrIOp::create(builder, loc,
                                         values.lookup(xorOp.getLhs()),
                                         values.lookup(xorOp.getRhs())));
        return success();
      })
      .Case([&](phys::AllFalseOp allFalse) { return lowerAllFalse(allFalse); })
      // Scheduling and transport do not change the ideal state, and they keep
      // each state's resource, so its wire stays where it is.
      .Case<phys::DelayOp, phys::BarrierOp, phys::MoveOp, qlx::event::FenceOp>(
          [&](Operation *) { return success(); })
      .Case([&](phys::ReturnOp ret) { return lowerReturn(ret); })
      .Case([&](qlx::event::SelectionOp selection) {
        return selection.emitOpError(
            "post-selects shots, which phys-to-quake does not support");
      })
      .Case<phys::ProduceResourceOp, phys::TransportResourceOp,
            phys::UnpackResourceOp, phys::PackResourceOp,
            phys::ResourceRequestOp, phys::ConsumeResourceOp,
            phys::DiscardResourcePayloadOp, phys::ResourceRotateProductOp,
            phys::FactoryStartOp>([&](Operation *flow) {
        return flow->emitOpError("is part of a resource or factory flow, "
                                 "which phys-to-quake does not support");
      })
      .Case([&](phys::EpochTransitionOp transition) {
        return transition.emitOpError(
            "changes the code epoch, and phys-to-quake cannot check that the "
            "operations before it realize the declared logical action");
      })
      .Default([&](Operation *other) -> LogicalResult {
        if (isa_and_present<arith::ArithDialect>(other->getDialect())) {
          builder.clone(*other, values);
          return success();
        }
        return other->emitOpError("has no lowering to Quake");
      });
}

LogicalResult GraphLowering::checkCarrier(Operation *op, Value state) {
  // The Phys verifier resolves every state's resource, and the resource's
  // class in the graph's machine.
  auto resourceName = cast<phys::StateType>(state.getType()).getResource();
  auto resource =
      symbols.lookupNearestSymbolFrom<phys::ResourceOp>(op, resourceName);
  auto resourceClass = cast<phys::ResourceClassOp>(symbols.lookupSymbolIn(
      machine, resource.getResourceClassAttr().getAttr()));
  // One state of a patch-granularity resource stands for a whole code patch,
  // so simulating it as one qubit would change the program.
  if (resourceClass.getGranularity() == "patch")
    return op->emitOpError()
           << "acquires resource " << resourceName
           << " of patch granularity, which phys-to-quake does not simulate";
  return success();
}

Attribute GraphLowering::carrierOf(Value state) {
  Attribute resource = cast<phys::StateType>(state.getType()).getResource();
  return aliases.lookup_or(resource, resource);
}

Value GraphLowering::wireOf(Value state) {
  return wires.lookup(carrierOf(state));
}

void GraphLowering::setWire(Value state, Value wire) {
  wires[carrierOf(state)] = wire;
}

LogicalResult GraphLowering::lowerAcquire(phys::AcquireOp op) {
  for (Value state : op.getStates()) {
    if (failed(checkCarrier(op, state)))
      return failure();
    setWire(state, quake::NullWireOp::create(
                       builder, op.getLoc(),
                       quake::WireType::get(builder.getContext())));
  }
  return success();
}

LogicalResult GraphLowering::lowerRelease(phys::ReleaseOp op) {
  for (Value state : op.getStates()) {
    quake::SinkOp::create(builder, op.getLoc(), wireOf(state));
    wires.erase(carrierOf(state));
  }
  return success();
}

/// Lowers `phys.prepare` and `phys.reset`, whose states mean the same thing.
LogicalResult GraphLowering::lowerPreparation(Operation *op, StringRef state,
                                              ValueRange inputs) {
  bool zero = state == "zero" || state == "0" || state == "loaded";
  bool plus = state == "plus" || state == "+";
  if (!zero && !plus)
    return op->emitOpError() << "prepares state '" << state
                             << "', which has no lowering to Quake";
  for (Value input : inputs) {
    Value wire = emitReset(builder, op->getLoc(), wireOf(input));
    if (plus)
      wire = emitH(builder, op->getLoc(), wire);
    setWire(input, wire);
  }
  return success();
}

LogicalResult GraphLowering::lowerApply(phys::ApplyOp op) {
  auto action = dyn_cast_or_null<phys::ActionOp>(
      resolveInMachine(symbols, machine, op, op.getActionAttr()));
  if (!action)
    return op.emitOpError() << "names action " << op.getActionAttr()
                            << ", which resolves to no phys.action";
  auto name = builtinProcess(action, action.getProcess());
  if (failed(name))
    return failure();
  // Reservations pass values that the process does not state, such as an
  // angle or a permutation.
  if (auto reservations = op.getReservationsAttr())
    return op.emitOpError() << "carries reservations " << reservations
                            << ", which have no lowering to Quake";
  // A broadcast action acts on each state alone. Otherwise each consecutive
  // group of `arity` states is one lane.
  std::size_t arity = action.getBroadcast() ? 1 : action.getArity();
  ValueRange inputs = op.getInputs();
  SmallVector<Value> inputWires;
  for (Value input : inputs)
    inputWires.push_back(wireOf(input));
  for (std::size_t lane = 0; lane < inputs.size(); lane += arity) {
    auto laneWires = ValueRange(inputWires).slice(lane, arity);
    auto outputs = emitBuiltinGate(builder, op.getLoc(), *name, laneWires);
    if (failed(outputs))
      return op.emitOpError()
             << "applies builtin process '" << *name << "' to " << arity
             << " states, which has no lowering to Quake";
    for (auto [i, wire] : llvm::enumerate(*outputs))
      setWire(inputs[lane + i], wire);
  }
  return success();
}

LogicalResult GraphLowering::lowerMeasure(phys::MeasureOp op) {
  auto instrument = dyn_cast_or_null<phys::InstrumentOp>(
      resolveInMachine(symbols, machine, op, op.getMeasurementAttr()));
  if (!instrument)
    return op.emitOpError() << "names instrument " << op.getMeasurementAttr()
                            << ", which resolves to no phys.instrument";
  // The process selects the basis. The Phys verifier ties the instrument's
  // kind to its process only for an instrument outside the machine.
  auto name = builtinProcess(instrument, instrument.getProcess());
  if (failed(name))
    return failure();
  bool xBasis = *name == "measure_x";
  if (!xBasis && *name != "measure_z")
    return op.emitOpError() << "measures with builtin process '" << *name
                            << "', which has no lowering to Quake";
  Location loc = op.getLoc();
  Value wire = wireOf(op.getInput());
  if (xBasis)
    wire = emitH(builder, loc, wire);
  auto *ctx = builder.getContext();
  auto mz = quake::MzOp::create(builder, loc, cc::MeasureHandleType::get(ctx),
                                TypeRange{quake::WireType::get(ctx)},
                                ValueRange{wire}, StringAttr{});
  values.map(op.getRecord(), mz.getMeasOut());
  wire = mz.getWires().front();
  if (op.getDestructive()) {
    quake::SinkOp::create(builder, loc, wire);
    wires.erase(carrierOf(op.getInput()));
    return success();
  }
  // Core's `quake.mx` does not rotate the state back, so an X measurement
  // that keeps its state is `h; mz; h`.
  if (xBasis)
    wire = emitH(builder, loc, wire);
  setWire(op.getInput(), wire);
  return success();
}

LogicalResult GraphLowering::lowerMeasureProduct(phys::MeasureProductOp op) {
  auto instrument = dyn_cast_or_null<phys::InstrumentOp>(
      resolveInMachine(symbols, machine, op, op.getInstrumentAttr()));
  if (!instrument)
    return op.emitOpError() << "names instrument " << op.getInstrumentAttr()
                            << ", which resolves to no phys.instrument";
  auto name = builtinProcess(instrument, instrument.getProcess());
  if (failed(name))
    return failure();
  if (*name != "measure_product")
    return op.emitOpError() << "measures with builtin process '" << *name
                            << "', which has no lowering to Quake";
  // The parity of the product collects on a fresh ancilla:
  //   ancilla = |0>; for each state: V_P; cx(state, ancilla); V_P^dagger
  //   mz(ancilla); sink(ancilla)
  // so every measured state keeps its post-measurement state.
  Location loc = op.getLoc();
  auto *ctx = builder.getContext();
  Value ancilla =
      quake::NullWireOp::create(builder, loc, quake::WireType::get(ctx));
  for (auto [input, pauli] : llvm::zip(op.getInputs(), op.getPaulis())) {
    StringRef label = cast<StringAttr>(pauli).getValue();
    Value wire =
        emitBasisChange(builder, loc, label, /*undo=*/false, wireOf(input));
    auto cx = emitGate<quake::XOp>(builder, loc, false, wire, ancilla);
    ancilla = cx[1];
    setWire(input, emitBasisChange(builder, loc, label, /*undo=*/true, cx[0]));
  }
  auto mz = quake::MzOp::create(builder, loc, cc::MeasureHandleType::get(ctx),
                                TypeRange{quake::WireType::get(ctx)},
                                ValueRange{ancilla}, StringAttr{});
  quake::SinkOp::create(builder, loc, mz.getWires().front());
  values.map(op.getRecord(), mz.getMeasOut());
  if (op.getInvert())
    invertedHandles.insert(mz.getMeasOut());
  return success();
}

/// Angles within this many quarter turns of a multiple of pi/2 are Clifford.
static constexpr double cliffordAngleTolerance = 1e-9;

LogicalResult GraphLowering::lowerRotateProduct(phys::RotateProductOp op) {
  // `exp(-i angle P / 2)`: change each Pauli to Z, collect the parity on the
  // last state with a CX ladder, rotate it about Z, and undo.
  Location loc = op.getLoc();
  SmallVector<Value> rotated;
  SmallVector<StringRef> labels;
  for (auto [input, pauli] : llvm::zip(op.getInputs(), op.getPaulis())) {
    labels.push_back(cast<StringAttr>(pauli).getValue());
    rotated.push_back(emitBasisChange(builder, loc, labels.back(),
                                      /*undo=*/false, wireOf(input)));
  }
  Value &target = rotated.back();
  for (Value &control : MutableArrayRef(rotated).drop_back()) {
    auto cx = emitGate<quake::XOp>(builder, loc, false, control, target);
    control = cx[0];
    target = cx[1];
  }
  // Stim runs no `rz`, so a Clifford angle becomes the Clifford gate that
  // equals `rz` up to a global phase.
  double angle = op.getAngle().convertToDouble();
  double quarterTurns = angle / (M_PI / 2);
  double nearest = std::round(quarterTurns);
  if (std::abs(quarterTurns - nearest) <= cliffordAngleTolerance) {
    switch ((static_cast<long long>(nearest) % 4 + 4) % 4) {
    case 1:
      target = emitS(builder, loc, /*adj=*/false, target);
      break;
    case 2:
      target = emitGate<quake::ZOp>(builder, loc, false, {}, target).front();
      break;
    case 3:
      target = emitS(builder, loc, /*adj=*/true, target);
      break;
    }
  } else {
    Value theta = arith::ConstantFloatOp::create(
        builder, loc, builder.getF64Type(), op.getAngle());
    target =
        emitGate<quake::RzOp>(builder, loc, false, {}, target, theta).front();
  }
  for (Value &control : llvm::reverse(MutableArrayRef(rotated).drop_back())) {
    auto cx = emitGate<quake::XOp>(builder, loc, false, control, target);
    control = cx[0];
    target = cx[1];
  }
  for (auto [input, label, wire] : llvm::zip(op.getInputs(), labels, rotated))
    setWire(input, emitBasisChange(builder, loc, label, /*undo=*/true, wire));
  return success();
}

/// Inlines a call body: its block arguments take `inputs`, and its yielded
/// values become `outputs`. States need no mapping, because a call keeps each
/// state's resource.
LogicalResult GraphLowering::lowerBody(phys::CallOp call, ValueRange inputs,
                                       ValueRange outputs) {
  Block &body = call.getBody().front();
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (!isa<phys::StateType>(argument.getType()))
      values.map(argument, values.lookup(input));
  if (failed(lowerOps(body)))
    return failure();
  auto yield = cast<phys::YieldOp>(body.getTerminator());
  for (auto [yielded, output] : llvm::zip(yield.getOperands(), outputs))
    if (!isa<phys::StateType>(yielded.getType()))
      values.map(output, values.lookup(yielded));
  return success();
}

LogicalResult GraphLowering::lowerCall(phys::CallOp op) {
  return lowerBody(op, op.getInputs(), op.getOutputs());
}

LogicalResult GraphLowering::lowerCallTemplate(phys::CallTemplateOp op) {
  phys::CallOp canonical = canonicalCalls.lookup(op.getTemplateEvent());
  // The body names the canonical call's resources. `state_aliases` renames
  // them to this invocation's carriers, and an enclosing replay renames those
  // in turn. `resource_substitutions` mirrors `state_aliases` exactly.
  DenseMap<Attribute, Attribute> enclosing = aliases;
  if (auto stateAliases = op.getStateAliasesAttr())
    for (auto alias : stateAliases.getAsRange<DictionaryAttr>()) {
      Attribute invocationResource = alias.get("alias");
      aliases[alias.get("template")] =
          enclosing.lookup_or(invocationResource, invocationResource);
    }
  // The elided form has an all-state boundary, so the body needs no mapping:
  // it acts on the carriers' current wires.
  LogicalResult lowered =
      op.getStateBoundaryElided()
          ? lowerBody(canonical, ValueRange{}, ValueRange{})
          : lowerBody(canonical, op.getInputs(), op.getOutputs());
  aliases = std::move(enclosing);
  return lowered;
}

LogicalResult GraphLowering::lowerOps(Block &block) {
  for (Operation &op : block.without_terminator())
    if (failed(lowerOp(&op)))
      return failure();
  return success();
}

/// The live carriers in resource-name order, so that the Quake is
/// deterministic.
SmallVector<Attribute> GraphLowering::liveCarriers() {
  SmallVector<Attribute> carriers(llvm::make_first_range(wires));
  llvm::sort(carriers, [](Attribute lhs, Attribute rhs) {
    return cast<FlatSymbolRefAttr>(lhs).getValue() <
           cast<FlatSymbolRefAttr>(rhs).getValue();
  });
  return carriers;
}

// Control flow carries every live carrier's wire through the Quake region,
// followed by the cflow op's classical values. A record crosses a region
// boundary as its `i1`, which `toBoolean` complements for an inverted product.
// States need no mapping inside a region, because each keeps its resource.
//
//   cflow.if %c -> (state<@a>, record)      %r:2 = cc.if(%c) ((%w = %a)) {
//     ...                              =>     ...
//     cflow.yield %a2, %m                     cc.continue %w2, %bit
//   } else { ... }                          } else { ... }

/// Maps a region's arguments: `arguments` holds the carriers' wires, then the
/// carried values of `physArguments`.
void GraphLowering::enterRegion(ValueRange arguments,
                                ArrayRef<Attribute> carriers,
                                ValueRange physArguments) {
  for (auto [carrier, wire] : llvm::zip(carriers, arguments))
    wires[carrier] = wire;
  ValueRange carried = arguments.drop_front(carriers.size());
  for (Value physArgument : physArguments)
    if (!isa<phys::StateType>(physArgument.getType())) {
      values.map(physArgument, carried.front());
      carried = carried.drop_front();
    }
}

/// The carriers' wires, then the carried values of `physValues`.
SmallVector<Value> GraphLowering::carriedValues(ArrayRef<Attribute> carriers,
                                                ValueRange physValues) {
  SmallVector<Value> carried;
  for (Attribute carrier : carriers)
    carried.push_back(wires.lookup(carrier));
  for (Value physValue : physValues) {
    if (isa<phys::StateType>(physValue.getType()))
      continue;
    carried.push_back(isa<phys::RecordType>(physValue.getType())
                          ? toBoolean(physValue)
                          : values.lookup(physValue));
  }
  return carried;
}

void GraphLowering::mapResults(ArrayRef<Attribute> carriers,
                               ValueRange physResults, ValueRange results) {
  wires.clear();
  for (auto [carrier, wire] : llvm::zip(carriers, results))
    wires[carrier] = wire;
  ValueRange carried = results.drop_front(carriers.size());
  for (Value physResult : physResults)
    if (!isa<phys::StateType>(physResult.getType())) {
      values.map(physResult, carried.front());
      carried = carried.drop_front();
    }
}

LogicalResult GraphLowering::lowerIf(qlx::cflow::IfOp op) {
  Location loc = op.getLoc();
  Value condition = values.lookup(op.getCondition());
  SmallVector<Attribute> carriers = liveCarriers();
  SmallVector<Value> linearArgs = carriedValues(carriers, {});
  // Mappings made in a branch do not dominate the code after the `cc.if`.
  IRMapping outerValues = values;
  DenseMap<Attribute, Value> outerWires = wires;
  // The result types follow from the branches, so they are lowered into a
  // scratch `cc.if` whose regions then move to the final one.
  auto noRegion = [](OpBuilder &, Location, Region &) {};
  auto scratch = cc::IfOp::create(builder, loc, TypeRange{}, condition,
                                  linearArgs, noRegion, noRegion);
  // The Phys verifier makes both branches consume and yield the same states,
  // so both end with the same carriers.
  SmallVector<Attribute> exitCarriers;
  SmallVector<Type> resultTypes;
  for (auto [source, target] :
       llvm::zip(op->getRegions(), scratch->getRegions())) {
    values = outerValues;
    wires = outerWires;
    Block *block = new Block();
    target.push_back(block);
    for (Value wire : linearArgs)
      block->addArgument(wire.getType(), loc);
    enterRegion(block->getArguments(), carriers, {});
    builder.setInsertionPointToStart(block);
    if (failed(lowerOps(source.front())))
      return failure();
    exitCarriers = liveCarriers();
    SmallVector<Value> continued = carriedValues(
        exitCarriers, source.front().getTerminator()->getOperands());
    resultTypes = llvm::to_vector(ValueRange(continued).getTypes());
    cc::ContinueOp::create(builder, loc, continued);
  }
  builder.setInsertionPoint(scratch);
  auto ifOp = cc::IfOp::create(
      builder, loc, resultTypes, condition, linearArgs,
      [&](OpBuilder &, Location, Region &region) {
        region.takeBody(scratch.getThenRegion());
      },
      [&](OpBuilder &, Location, Region &region) {
        region.takeBody(scratch.getElseRegion());
      });
  scratch.erase();
  builder.setInsertionPointAfter(ifOp);
  values = std::move(outerValues);
  mapResults(exitCarriers, op.getResults(), ifOp.getResults());
  return success();
}

LogicalResult GraphLowering::lowerRepeat(qlx::cflow::RepeatOp op) {
  Location loc = op.getLoc();
  SmallVector<Attribute> carriers = liveCarriers();
  Value count = arith::ConstantIntOp::create(builder, loc, op.getCount(), 64);
  SmallVector<Value> iterArgs{
      arith::ConstantIntOp::create(builder, loc, 0, 64)};
  llvm::append_range(iterArgs, carriedValues(carriers, op.getInits()));
  auto types = llvm::to_vector(ValueRange(iterArgs).getTypes());
  IRMapping outerValues = values;
  Block &body = op.getBody().front();
  LogicalResult lowered = success();
  // The Phys verifier makes the body yield every state it receives, so each
  // iteration ends with the carriers it began with.
  auto loop = cc::LoopOp::create(
      builder, loc, types, iterArgs, /*postCond=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        Value more =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                  block.getArgument(0), count);
        cc::ConditionOp::create(builder, loc, more, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        enterRegion(block.getArguments().drop_front(), carriers,
                    body.getArguments());
        lowered = lowerOps(body);
        if (failed(lowered))
          return;
        SmallVector<Value> continued{block.getArgument(0)};
        llvm::append_range(
            continued,
            carriedValues(carriers, body.getTerminator()->getOperands()));
        cc::ContinueOp::create(builder, loc, continued);
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        SmallVector<Value> next(block.getArguments());
        next.front() = arith::AddIOp::create(
            builder, loc, next.front(),
            arith::ConstantIntOp::create(builder, loc, 1, 64));
        cc::ContinueOp::create(builder, loc, next);
      });
  if (failed(lowered))
    return failure();
  values = std::move(outerValues);
  mapResults(carriers, op.getResults(), loop.getResults().drop_front());
  return success();
}

LogicalResult GraphLowering::lowerWhile(qlx::cflow::WhileOp op) {
  if (op.getMaxIterationsAttr())
    return op.emitOpError(
        "bounds the loop by max_iterations, and Logical does not define what "
        "happens at the bound");
  Location loc = op.getLoc();
  SmallVector<Attribute> carriers = liveCarriers();
  SmallVector<Value> iterArgs = carriedValues(carriers, op.getInits());
  auto types = llvm::to_vector(ValueRange(iterArgs).getTypes());
  IRMapping outerValues = values;
  Block &before = op.getBeforeRegion().front();
  Block &after = op.getAfterRegion().front();
  LogicalResult lowered = success();
  // Each region starts from the outer mappings: a Boolean read in the
  // condition does not dominate the body.
  auto loop = cc::LoopOp::create(
      builder, loc, types, iterArgs, /*postCond=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        values = outerValues;
        enterRegion(block.getArguments(), carriers, before.getArguments());
        lowered = lowerOps(before);
        if (failed(lowered))
          return;
        auto condition =
            cast<qlx::cflow::WhileConditionOp>(before.getTerminator());
        cc::ConditionOp::create(
            builder, loc, values.lookup(condition.getCondition()),
            carriedValues(carriers, condition.getForwarded()));
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        if (failed(lowered))
          return;
        values = outerValues;
        enterRegion(block.getArguments(), carriers, after.getArguments());
        lowered = lowerOps(after);
        if (failed(lowered))
          return;
        cc::ContinueOp::create(
            builder, loc,
            carriedValues(carriers, after.getTerminator()->getOperands()));
      });
  if (failed(lowered))
    return failure();
  values = std::move(outerValues);
  mapResults(carriers, op.getResults(), loop.getResults());
  return success();
}

/// Records are not linear, so a record that several ops read is discriminated
/// once, at its first read.
Value GraphLowering::toBoolean(Value physValue) {
  Value value = values.lookup(physValue);
  if (!isa<cc::MeasureHandleType>(value.getType()))
    return value;
  Location loc = physValue.getLoc();
  Value bit =
      quake::DiscriminateOp::create(builder, loc, builder.getI1Type(), value);
  if (invertedHandles.contains(value))
    bit = arith::XOrIOp::create(
        builder, loc, bit, arith::ConstantIntOp::create(builder, loc, 1, 1));
  values.map(physValue, bit);
  return bit;
}

LogicalResult GraphLowering::lowerCondition(phys::ConditionOp op) {
  Value bit = toBoolean(op.getSource());
  if (!op.getExpected()) {
    Value one = arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
    bit = arith::XOrIOp::create(builder, op.getLoc(), bit, one);
  }
  values.map(op.getResult(), bit);
  return success();
}

LogicalResult GraphLowering::lowerAllFalse(phys::AllFalseOp op) {
  Location loc = op.getLoc();
  Value any = arith::ConstantIntOp::create(builder, loc, 0, 1);
  for (Value event : op.getEvents())
    any = arith::OrIOp::create(builder, loc, any, values.lookup(event));
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
  values.map(op.getResult(), arith::XOrIOp::create(builder, loc, any, one));
  return success();
}

LogicalResult GraphLowering::lowerReturn(phys::ReturnOp op) {
  Location loc = op.getLoc();
  SmallVector<Value> bits;
  for (Value operand : op.getOperands())
    bits.push_back(toBoolean(operand));
  if (bits.size() <= 1) {
    func::ReturnOp::create(builder, loc, bits);
    return success();
  }
  // Two or more Booleans return as one list, in heap storage that the caller
  // frees, as both CUDA-Q front-ends return a list.
  // Note: Keep the `!cc.array<i1 x N>` buffer and its constant length. Core
  // folds that length through to `ReturnToOutputLog`, which records the list
  // on QIR profiles. A buffer cast to `!cc.ptr<i1>` loses the length there,
  // and profile output then drops the list without a diagnostic.
  auto i1Type = builder.getI1Type();
  Value buffer = cc::AllocaOp::create(
      builder, loc,
      cc::ArrayType::get(builder.getContext(), i1Type, bits.size()));
  auto slotType = cc::PointerType::get(i1Type);
  for (auto [i, bit] : llvm::enumerate(bits)) {
    Value slot = cc::ComputePtrOp::create(
        builder, loc, slotType, buffer,
        ArrayRef<cc::ComputePtrArg>{static_cast<std::int32_t>(i)});
    cc::StoreOp::create(builder, loc, bit, slot);
  }
  Value length = arith::ConstantIntOp::create(builder, loc, bits.size(), 64);
  Value list = cc::SequenceInitOp::create(
      builder, loc, cc::SequenceType::get(i1Type), buffer, length);
  func::ReturnOp::create(
      builder, loc,
      cudaq::opt::marshal::copyDynamicValueToHeap(loc, builder, list));
  return success();
}

namespace {

class PhysToQuakePass : public qlx::impl::PhysToQuakeBase<PhysToQuakePass> {
public:
  using PhysToQuakeBase::PhysToQuakeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto graph = dyn_cast_or_null<phys::GraphOp>(
        SymbolTable::lookupSymbolIn(module, entry));
    if (entry.empty() || !graph) {
      module.emitError() << "phys-to-quake needs the entry option to name a "
                            "phys.graph, but '"
                         << entry << "' names none";
      return signalPassFailure();
    }
    SmallVector<Operation *> source;
    for (Operation &op : *module.getBody())
      source.push_back(&op);
    auto entryPoint = GraphLowering(module, graph).lower();
    if (failed(entryPoint))
      return signalPassFailure();
    // The module now holds Quake only: the entry point and the intrinsic that
    // copies a list result to the heap. Loading it after the erasure keeps a
    // source symbol of the same name from hiding it.
    for (Operation *op : llvm::reverse(source))
      op->erase();
    module.push_back(*entryPoint);
    if (llvm::any_of(entryPoint->getResultTypes(),
                     llvm::IsaPred<cc::SequenceType>) &&
        failed(cudaq::IRBuilder::atBlockEnd(module.getBody())
                   .loadIntrinsic(module, "__nvqpp_vectorCopyCtor")))
      return signalPassFailure();
    SmallVector<StringAttr> logicalAttrs;
    for (NamedAttribute attr : module->getDiscardableAttrs())
      if (attr.getName().getValue().starts_with("qlx."))
        logicalAttrs.push_back(attr.getName());
    for (StringAttr name : logicalAttrs)
      module->removeAttr(name);
    module->setAttr(cudaq::runtime::disableQuantumOpts,
                    UnitAttr::get(module.getContext()));
  }
};

} // namespace
