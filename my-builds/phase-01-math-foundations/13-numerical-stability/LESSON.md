# 13 — Numerical Stability

> Computers round every number; good code keeps the rounding from snowballing.

**Project:** A tour of floating-point failure modes built from scratch: broken decimal math, overflowing softmax, catastrophic cancellation, the finite-difference error U-shape, and a float16 vs bfloat16 showdown — each paired with the fix that makes it safe.

## What I built
- 0.1 + 0.2 != 0.3 demo and a loop that discovers machine epsilon by halving
- float32 probe: max value, smallest value, epsilon, and the growing gaps between neighbors
- Naive softmax that turns into NaN on big logits, next to the stable subtract-the-max version
- Log-sum-exp trick: same idea applied to log(sum(exp(x)))
- Catastrophic cancellation: sqrt(x+1) - sqrt(x) losing 8 digits, and the rewrite that saves them
- Finite-difference step-size sweep showing the error U-shape and the best step near 1e-5
- float16 vs simulated bfloat16 comparison table (range vs precision)

## Main points learned
- Floats store numbers in binary, and 0.1 has no exact binary form. Tiny errors are normal.
- Machine epsilon (~2.2e-16 for float64) is the smallest nudge that still changes 1.0.
- Float precision is relative: near 1.0 the gaps are tiny, near 1e8 the gap is 8 whole units.
- exp() of a big number overflows to infinity. Subtracting the max first costs nothing and fixes it.
- Subtracting two nearly equal numbers erases their leading digits. That is catastrophic cancellation.
- Finite-difference steps have a sweet spot: too big is a crude formula, too small is cancellation.
- bfloat16 trades precision for float32's range. Training survives coarse numbers but not infinities.

## The algorithms, explained simply
**Machine epsilon discovery.** Keep halving a small number until adding it to 1.0 changes nothing. It is like whispering quieter and quieter until your friend can no longer hear you — that threshold is the computer's hearing limit.

**Stable softmax.** Before exponentiating scores, subtract the biggest one so the largest becomes 0. Like converting race times to "seconds behind the leader": the rankings are identical, but the numbers stay small enough to work with.

**Log-sum-exp trick.** Pull the biggest value out front, do the exp/sum/log on the small leftovers, then add the big value back at the end. Same answer, but the dangerous huge number never gets exponentiated.

**Finite-difference sweep.** Estimate a slope by checking the function a step to each side, and try many step sizes. A giant step misreads a curve as a line; a microscopic step makes two floats so similar their difference is mostly noise. The best step sits in the valley between the two failure modes.

**bfloat16 simulation.** Chop a float32's detail bits but keep its exponent bits. It is a measuring tape that still reaches a mile but only marks whole inches — short reach (float16) kills training with overflow; coarse markings (bfloat16) mostly wash out.

## How this shows up in AI
- Every real softmax and cross-entropy implementation uses the subtract-max / log-sum-exp trick; without it, attention and loss computations NaN out.
- Mixed-precision training defaults to bfloat16 because gradients and activations spike, and range beats precision for not exploding.
- Gradient checking uses centered finite differences, and picking h ~ 1e-5 is exactly the U-shape lesson.
- Loss curves that suddenly go to NaN usually trace back to one of these failure modes: overflow, underflow, or cancellation.

## Run it
```
cd 13-numerical-stability
python3 project.py
```
