/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

/// \file
/// Some working definitions:
///
/// A \em counted loop: a loop that counts from $0$ up to $n-1$ stepping by $1$.
/// Such a loop is \em normalized (starts at $0$), \em monotonically increasing
/// (slope is a constant $1$), executes exactly $n$ times, and $n$ is a
/// compile-time constant. A counted loop is said to have static control flow.
///
/// A \em{constant upper bound} loop: a loop that counts from $0$ up to $m$
/// where $m <= n-1$ stepping by $1$.
///
/// An \em invariant loop: a counted loop but $n$ need not be a compile-time
/// constant. An invariant loop cannot be fully unrolled until runtime. In
/// quantum circuit speak, one does not know the full size of the circuit.
///
/// A \em monotonic loop: a loop that counts from $i$ up to (down to) $j$
/// stepping by positive (negative) integral values; mathematically, it is a
/// strictly monotonic sequence. If the step is a compile-time constant, $k$,
/// then a closed iterval monotonic loop must execute exactly $\max(0, \floor{(j
/// - i + k) / k})$ iterations. By normalizing a monotonic loop and constant
/// folding and propagation, we may be able to convert it to static control
/// flow.
///
/// For completeness, a \em{conditionally iterated} loop is a monotonic loop
/// that has a second auxilliary condition to determine if a given loop
/// iteration is executed or not. (A constant upper bound loop, see above, is a
/// subclass of a conditionally iterated loop.) For example, the condition might
/// be used in iteration $m$ to disable all subsequent iterations. (Much like a
/// `break` statement.) Another example would be a condition that disables all
/// the even iterations. These loops might be unrolled but only if the loop can
/// be normalized into static control flow. It is helpful in pruning the amount
/// of unrolling if the auxillary condition can be computed as a constant. It is
/// likely these loops cannot be converted to static control flow and would thus
/// need to be expanded at runtime.

static Value peelCastOps(Value v) {
  Operation *defOp = nullptr;
  for (; (defOp = v.getDefiningOp());) {
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
            cudaq::cc::CastOp>(defOp))
      v = defOp->getOperand(0);
    else
      break;
  }
  return v;
}

static bool isaConstant(Value v) {
  v = peelCastOps(v);
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    return isa<IntegerAttr>(c.getValue());
  return false;
}

static bool isaConstantOf(Value v, std::int64_t hasVal) {
  v = peelCastOps(v);
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt() == hasVal;
  return false;
}

static bool isNegativeConstant(Value v) {
  v = peelCastOps(v);
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt() < 0;
  return false;
}

static bool isClosedIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ule || p == arith::CmpIPredicate::sle;
}

static bool isSemiOpenIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::slt ||
         p == arith::CmpIPredicate::ne;
}

static bool validCountedLoopIntervalForm(arith::CmpIOp cmp,
                                         bool allowClosedInterval) {
  auto p = cmp.getPredicate();
  return isSemiOpenIntervalForm(p) ||
         (allowClosedInterval && isClosedIntervalForm(p));
}

// If the value, v, dominates the loop then it is invariant by definition. Block
// arguments that are, in fact, a threaded invariant value should have been
// converted to their dominating definition by the canonicalization pass.
static bool isLoopInvariant(ArrayRef<Value> vs, cudaq::cc::LoopOp loop) {
  DominanceInfo dom(loop->getParentOfType<func::FuncOp>());
  for (auto v : vs)
    if (!dom.dominates(v, loop.getOperation()))
      return false;
  return true;
}

/// Returns a pair `(true, stepValue)` if and only if the operation, \p op, is
/// an induction computation (integer add or subtract). Otherwise returns
/// `(false, null)`.
static std::pair<bool, Value> isInductionOn(unsigned offset, Operation *op,
                                            ArrayRef<BlockArgument> args) {
  if (auto addOp = dyn_cast_or_null<arith::AddIOp>(op)) {
    if (addOp.getLhs() == args[offset])
      return {true, addOp.getRhs()};
    if (addOp.getRhs() == args[offset])
      return {true, addOp.getLhs()};
  } else if (auto subOp = dyn_cast_or_null<arith::SubIOp>(op)) {
    if (subOp.getLhs() == args[offset])
      return {true, subOp.getRhs()};
  }
  return {false, Value{}};
}

// TODO: consider caching the results.
static BlockArgument getLinearExpr(Value expr,
                                   cudaq::opt::LoopComponents &result,
                                   cudaq::cc::LoopOp loop) {
  auto v = peelCastOps(expr);
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    // Trivial expression: bare argument.
    return ba;
  }
  auto checkAndSet = [&](Value va, Value vb, Value &saved) -> BlockArgument {
    auto vl = peelCastOps(va);
    if (auto ba = dyn_cast<BlockArgument>(vl);
        ba && isLoopInvariant(vb, loop)) {
      saved = vb;
      return ba;
    }
    return {};
  };
  auto scaledIteration = [&](Value v) -> BlockArgument {
    if (auto mulOp = v.getDefiningOp<arith::MulIOp>()) {
      result.reciprocalScale = false;
      if (auto ba =
              checkAndSet(mulOp.getLhs(), mulOp.getRhs(), result.scaleValue))
        return ba;
      return checkAndSet(mulOp.getRhs(), mulOp.getLhs(), result.scaleValue);
    }
    if (auto divOp = v.getDefiningOp<arith::DivUIOp>()) {
      result.reciprocalScale = true;
      return checkAndSet(divOp.getLhs(), divOp.getRhs(), result.scaleValue);
    }
    if (auto divOp = v.getDefiningOp<arith::DivSIOp>()) {
      result.reciprocalScale = true;
      return checkAndSet(divOp.getLhs(), divOp.getRhs(), result.scaleValue);
    }
    return {};
  };
  if (auto addOp = expr.getDefiningOp<arith::AddIOp>()) {
    result.negatedAddend = false;
    result.minusOneMult = false;
    if (auto ba =
            checkAndSet(addOp.getLhs(), addOp.getRhs(), result.addendValue))
      return ba;
    if (auto ba = scaledIteration(addOp.getLhs());
        ba && isLoopInvariant({addOp.getRhs()}, loop)) {
      result.addendValue = addOp.getRhs();
      return ba;
    }
    if (auto ba =
            checkAndSet(addOp.getRhs(), addOp.getLhs(), result.addendValue))
      return ba;
    if (auto ba = scaledIteration(addOp.getRhs());
        ba && isLoopInvariant({addOp.getLhs()}, loop)) {
      result.addendValue = addOp.getLhs();
      return ba;
    }
    return {};
  }
  if (auto subOp = expr.getDefiningOp<arith::SubIOp>()) {
    if (auto ba =
            checkAndSet(subOp.getLhs(), subOp.getRhs(), result.addendValue)) {
      result.negatedAddend = true;
      return ba;
    }
    if (auto ba = scaledIteration(subOp.getLhs());
        ba && isLoopInvariant({subOp.getRhs()}, loop)) {
      result.addendValue = subOp.getRhs();
      result.negatedAddend = true;
      return ba;
    }
    if (auto ba =
            checkAndSet(subOp.getRhs(), subOp.getLhs(), result.addendValue)) {
      result.minusOneMult = true;
      return ba;
    }
    if (auto ba = scaledIteration(subOp.getRhs());
        ba && isLoopInvariant({subOp.getLhs()}, loop)) {
      result.addendValue = subOp.getLhs();
      result.minusOneMult = true;
      return ba;
    }
    return {};
  }
  return scaledIteration(expr);
}

namespace cudaq {

bool opt::isSemiOpenPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::slt ||
         p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::sgt ||
         p == arith::CmpIPredicate::ne;
}

bool opt::isUnsignedPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::ule ||
         p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::uge;
}

// We expect the loop control value to have the following form.
//
//   %final = cc.loop while ((%iter = %initial) -> (iN)) {
//     ...
//     %cond = arith.cmpi {<.<=,!=,>=,>}, %iter, %bound : iN
//     cc.condition %cond (%iter : iN)
//   } do {
//    ^bb1(%iter : iN):
//     ...
//     cc.continue %iter : iN
//   } step {
//    ^bb2(%iter : iN):
//     ...
//     %next = arith.{addi,subi} %iter, %step : iN
//     cc.continue %next : iN
//   }
//
// with the additional requirement that none of the `...` sections can modify
// the value of `%bound` or `%step`. Those values are invariant if there are
// no side-effects in the loop Op (no store or call operations) and these values
// do not depend on a block argument.
bool opt::hasMonotonicControlInduction(cc::LoopOp loop, LoopComponents *lcp) {
  if (loop.getInitialArgs().empty() || loop.getResults().empty())
    return false;
  if (auto c = getLoopComponents(loop)) {
    if (lcp)
      *lcp = *c;
    if (isLoopInvariant({c->compareValue, c->stepValue}, loop))
      return (bool)getLinearExpr(c->getCompareInduction(), *c, loop);
  }
  return false;
}

static bool allExitsAreContinue(Region &reg) {
  for (auto &block : reg)
    if (block.hasNoSuccessors() && !isa<cc::ContinueOp>(block.getTerminator()))
      return false;
  return true;
}

bool opt::loopContainsBreak(cc::LoopOp loopOp) {
  return !allExitsAreContinue(loopOp.getBodyRegion());
}

bool opt::isaMonotonicLoop(Operation *op, bool allowEarlyExit,
                           LoopComponents *lcp) {
  if (auto loopOp = dyn_cast_or_null<cc::LoopOp>(op)) {
    // Cannot be a `do while` loop. See cc-loop-peeling.
    if (loopOp.isPostConditional())
      return false;
    auto &reg = loopOp.getBodyRegion();
    return !reg.empty() && (allowEarlyExit || allExitsAreContinue(reg)) &&
           hasMonotonicControlInduction(loopOp, lcp);
  }
  return false;
}

bool opt::isaInvariantLoop(const LoopComponents &c, bool allowClosedInterval) {
  if (isaConstantOf(c.initialValue, 0) && isaConstantOf(c.stepValue, 1) &&
      isa<arith::AddIOp>(c.stepOp) && !isNegativeConstant(c.compareValue)) {
    auto cmp = cast<arith::CmpIOp>(c.compareOp);
    return validCountedLoopIntervalForm(cmp, allowClosedInterval);
  }
  return false;
}

bool opt::isaInvariantLoop(cc::LoopOp loop, bool allowClosedInterval,
                           bool allowEarlyExit, LoopComponents *lcp) {
  LoopComponents c;
  if (isaMonotonicLoop(loop.getOperation(), allowEarlyExit, &c)) {
    if (lcp)
      *lcp = c;
    return isaInvariantLoop(c, allowClosedInterval);
  }
  return false;
}

bool opt::isaCountedLoop(cc::LoopOp loop, bool allowClosedInterval) {
  LoopComponents c;
  return isaInvariantLoop(loop, allowClosedInterval, /*allowEarlyExit=*/false,
                          &c) &&
         isaConstant(c.compareValue);
}

bool opt::isaConstantUpperBoundLoop(cc::LoopOp loop, bool allowClosedInterval) {
  LoopComponents c;
  return isaInvariantLoop(loop, allowClosedInterval, /*allowEarlyExit=*/true,
                          &c) &&
         isaConstant(c.compareValue);
}

Value opt::LoopComponents::getCompareInduction() {
  auto cmpOp = cast<arith::CmpIOp>(compareOp);
  return cmpOp.getLhs() == compareValue ? cmpOp.getRhs() : cmpOp.getLhs();
}

bool opt::LoopComponents::stepIsAnAddOp() { return isa<arith::AddIOp>(stepOp); }

bool opt::LoopComponents::shouldCommuteStepOp() {
  if (auto addOp = dyn_cast_or_null<arith::AddIOp>(stepOp))
    return addOp.getRhs() == stepRegion->front().getArgument(induction);
  // Note: we don't allow induction on lhs of subtraction.
  return false;
}

bool opt::LoopComponents::isClosedIntervalForm() {
  auto cmp = cast<arith::CmpIOp>(compareOp);
  return ::isClosedIntervalForm(cmp.getPredicate());
}

bool opt::LoopComponents::isLinearExpr() { return addendValue || scaleValue; }

template <typename T>
constexpr int computeArgsOffset() {
  if constexpr (std::is_same_v<T, cc::ConditionOp>) {
    return 1;
  } else {
    return 0;
  }
}

std::optional<opt::LoopComponents> opt::getLoopComponents(cc::LoopOp loop) {
  opt::LoopComponents result;
  auto &whileRegion = loop.getWhileRegion();
  auto &whileEntry = whileRegion.front();
  auto condOp = cast<cc::ConditionOp>(whileRegion.back().back());
  result.compareOp = condOp.getCondition().getDefiningOp();
  auto cmpOp = cast<arith::CmpIOp>(result.compareOp);

  auto argumentToCompare = [&](unsigned idx) -> bool {
    return (getLinearExpr(cmpOp.getLhs(), result, loop) ==
            whileEntry.getArgument(idx)) ||
           (getLinearExpr(cmpOp.getRhs(), result, loop) ==
            whileEntry.getArgument(idx));
  };
  auto scanRegionForStep = [&]<typename TERM,
                               int argsOff = computeArgsOffset<TERM>()>(Region &
                                                                        reg)
                               ->std::optional<unsigned> {
    // Pre-scan to make sure all terminators are ContinueOp.
    for (auto &block : reg)
      if (block.hasNoSuccessors())
        if (!isa<TERM>(block.back()))
          return {};

    for (auto &block : reg) {
      if (block.hasNoSuccessors()) {
        if (auto contOp = cast<TERM>(block.back())) {
          // Find an argument to the ContinueOp that is an integral induction
          // and updated by a step value.
          for (auto pr :
               llvm::enumerate(contOp.getOperands().drop_front(argsOff))) {
            if (auto *defOp = pr.value().getDefiningOp()) {
              if ((defOp->getBlock() == &block) &&
                  isa<arith::AddIOp, arith::SubIOp>(defOp)) {
                auto ps = isInductionOn(pr.index(), defOp,
                                        reg.front().getArguments());
                if (ps.first && argumentToCompare(pr.index())) {
                  // Set the step value and step op here.
                  result.stepValue = ps.second;
                  result.stepOp = defOp;
                  result.stepRegion = &reg;
                  return pr.index();
                }
              }
            }
          }
        }
      }
    }
    return {};
  };

  if (loop.hasStep()) {
    // Loop has a step region, so look for the step op.
    // as in: `for (i = 0; i < n; i++) ...`
    if (auto stepPosOpt = scanRegionForStep.template operator()<cc::ContinueOp>(
            loop.getStepRegion()))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp) {
    // If step has not been found, look in the body region.
    // as in: `for (i = 0; i < n;) { ... i++; }`
    if (auto stepPosOpt = scanRegionForStep.template operator()<cc::ContinueOp>(
            loop.getBodyRegion()))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp) {
    // If step has still not been found, look in the while region.
    // as in: `for (i = n; i-- > 0;) ...`
    if (auto stepPosOpt =
            scanRegionForStep.template operator()<cc::ConditionOp>(whileRegion))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp)
    return {};

  result.initialValue = loop.getInitialArgs()[result.induction];

  // The comparison operation allows for the induction value to appear as part
  // of a loop-invariant linear expression on one side of the comparison. This
  // allows for invariant expressions on each side, such as, `4 * i + 1 < exp`.
  // This relaxation to invariant expressions requires some transformations to
  // normalize the comparison operation. Taking the example, this would
  // transform to `i < (exp - 1) / 4`.
  // TODO: A possible extension is to detect \em{conditionally iterated} loops
  // and open those up to further analysis and transformations such as loop
  // unrolling.
  if (getLinearExpr(cmpOp.getLhs(), result, loop) ==
      whileEntry.getArgument(result.induction))
    result.compareValue = cmpOp.getRhs();
  else if (getLinearExpr(cmpOp.getRhs(), result, loop) ==
           whileEntry.getArgument(result.induction))
    result.compareValue = cmpOp.getLhs();
  else
    return {};
  return result;
}

} // namespace cudaq
