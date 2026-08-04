# Course Feedback — 17-linear-systems

**Reviewed:** code/linear_systems.py (ran, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, least-squares worked example (~line 252).** The example solves the normal equations `[[4,10],[10,30]] x = [22, 63]` and states "Solve: x = [1.5, 1.7]". The correct solution is **[1.5, 1.6]** (verified by hand and with numpy: 4(1.5) + 10(1.6) = 22, 10(1.5) + 30(1.6) = 63). The slope 1.7 is wrong.

## Nitpicks & suggestions

- The partial-pivoting example (docs, "Partial pivoting: why it matters") shows both the pivoted AND unpivoted computations arriving at the correct answer ("x1 = 0.001/0.001 = 1.000 (correct)"), so it never demonstrates the failure it warns about. The classic version of this example needs explicit limited-precision rounding (e.g. 3 significant digits) to show the unpivoted answer going wrong.
- The conjugate-gradient demo runs all 50 iterations on the kappa ≈ 900 system and stops with residual 4.4e-2 — i.e. it did NOT converge to tol=1e-10 within n steps. The docs correctly say "in exact arithmetic," but the demo output silently contradicts the "at most n iterations" story; one printed line explaining the floating-point caveat would help.
- `ridge_regression` penalizes the intercept column along with the features. This is consistent with the sklearn comparison (which uses `fit_intercept=False`), but it is non-standard practice; worth a note so students don't copy it blindly.
- `condition_number` computes a full SVD just for the ratio; fine here, but a comment that `np.linalg.cond` exists would be nice.

## What's solid

- All from-scratch solvers are correct and verified in the demos against numpy/sklearn to ~1e-16: Gaussian elimination with partial pivoting, LU (PA = LU with correct L row-swapping, a detail many from-scratch versions get wrong), Cholesky, normal equations, ridge, and CG.
- The conceptual docs are strong: column vs row picture, the LU amortization argument, condition-number rules of thumb, and the which-method-when table are accurate; the quiz answers (including the digits-of-precision question) are all correct.
