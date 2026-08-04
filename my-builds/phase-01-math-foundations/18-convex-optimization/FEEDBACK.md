# Course Feedback — 18-convex-optimization

**Reviewed:** code/convex.py (ran with python3, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Clean

## Bugs & errors

None found.

- convex.py runs cleanly. All demos produce correct results: the convexity checker classifies all 10 test functions right, Hessian eigenvalues are correct (e.g. [[2,3],[3,2]] gives 5 and -1, correctly reported as indefinite), Newton converges in 1 step on the quadratic, and both Lagrange problems converge to the true analytical solutions (x=y=0.5, lambda=-1; and x=2, y=1, lambda=2 — I recomputed both by hand).
- The duality demo's algebra checks out: d(lambda) = lambda - lambda^2/2, lambda* = 1, primal = dual = 0.5.
- The L1/L2 regularization geometry demo is right. The disk-constrained minimum really is the radial projection (3,2)/sqrt(13), and the diamond-constrained minimum really is the corner (1,0) with objective 8.
- All 5 quiz answers and explanations are correct.

## Nitpicks & suggestions

1. docs/en.md, Step 2 snippet: the Newton function checks convergence (`sum(gi**2) < tol`) after the update, using the gradient from before the step, and takes an unused `f` parameter. The actual convex.py checks before stepping. Harmless, but the doc snippet does one extra iteration.
2. docs/en.md, Step 4: says gradient descent "will take hundreds of steps" for a condition number of 5. With a well-tuned learning rate the code's own condition-number demo shows 85 steps at c=5. "Hundreds" only holds for higher condition numbers or worse learning rates.
3. docs/en.md, "Diagonal approximations": calling Adam's second moment "a diagonal approximation of the Hessian's diagonal" is loose. It is an estimate of the squared gradient (closer to the Fisher/empirical Fisher diagonal), not the Hessian.

## What's solid

- Every numeric claim in the demos is verifiable in the output, and the Lagrange/duality worked examples match the analytical solutions exactly.
- The "why deep learning works despite non-convexity" section is accurate and well-sourced (saddle points, overparameterization, flat minima).
