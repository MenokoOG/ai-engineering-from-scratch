# Course Feedback — 08 Optimization

**Reviewed:** code/optimizers.py (ran with python3), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Build It" Step 5 (`optimize` snippet) + Exercise 1.** The docs' `optimize()` has no divergence guard, unlike the real code in optimizers.py (which checks for nan/inf and catches OverflowError). Exercise 1 tells the student to sweep learning rates up to 0.01 on Rosenbrock. Verified: running the docs snippet with lr=0.005 crashes with `OverflowError: (34, 'Numerical result out of range')` instead of showing divergence. Fix: add the nan/inf guard (or try/except) to the Step 5 snippet, or warn that large learning rates will overflow.

2. **code/optimizers.py, `demo_saddle_point` (lines ~262-280) + docs Exercise 3.** The demo labels vanilla GD as "Escaped? no", but that is an artifact of the 200-step cap and the |y| > 1.0 threshold. On f = x^2 - y^2 from (0.01, 0.01), plain GD grows y by factor 1.02 per step and passes |y| = 1 at step ~233. Verified by running it longer. So GD does escape this saddle; it is just slower. The framing "which escapes the saddle point?" teaches a slightly wrong conclusion (GD only stalls exactly on the stable manifold, i.e., y = 0). Fix: say GD escapes "much more slowly", or start at (0.01, 0.0) to show true stalling.

## Nitpicks & suggestions

1. In the saddle demo, SGD+Momentum ends at y = 4.7 million and f = -2.2e13. Technically "escaped", but the demo is really showing divergence on an unbounded function. A cap or comment would make the output less alarming.
2. "Ship It" references `outputs/prompt-optimizer-guide.md`, which does not exist in the lesson folder.
3. The learning-rate mermaid diagram uses lr = 0.0001 as "too small", but the actual demos use lr = 0.0005 and 0.0001 as working values for this problem. Slight mixed message.
4. "A convex function has one minimum. Gradient descent always finds it." Overstated: convex functions can have a flat set of minima or none at all (e.g., e^x), and GD still needs a sane learning rate. Fine for intuition, not literally true.

## What's solid

- The three optimizer implementations (GD, momentum, Adam with bias correction) are textbook-correct; I verified the Rosenbrock gradient and Adam update against the standard formulas.
- The side-by-side convergence comparison and lr/momentum sweeps give exactly the right intuition, and the real code handles divergence gracefully.
