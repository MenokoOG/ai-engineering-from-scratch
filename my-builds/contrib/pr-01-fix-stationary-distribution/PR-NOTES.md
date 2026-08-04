# PR #1 prep — fix stationary_distribution in Phase 1, Lesson 22

**Status:** tested, ready for your review. Rewrite the description below in your own words before submitting.

## The bug

File: `phases/01-math-foundations/22-stochastic-processes/code/stochastic.py`, `MarkovChain.stationary_distribution()`.

`np.linalg.eig` returns eigenvectors with arbitrary sign. For the lesson's weather matrix it returns the eigenvalue-1 eigenvector with all-negative components. The code then runs `np.clip(stationary, 0, None)`, which zeroes every component. The sum is 0, the normalize branch is skipped, and the function returns `[0, 0, 0]`.

Two visible symptoms when you run the file as-is:

1. "Stationary distribution (analytical)" prints all zeros.
2. The convergence check compares the empirical distribution against those zeros, so it prints max error ~0.55 at every chain length and never converges.

## The fix (4 lines → 1 line)

```diff
         stationary = np.real(eigenvectors[:, idx])
-        stationary = np.clip(stationary, 0, None)
-        total = stationary.sum()
-        if total > 0:
-            stationary = stationary / total
+        stationary = stationary / stationary.sum()
         return stationary
```

Why this is correct: for a stochastic matrix, the eigenvalue-1 eigenvector of P.T has all components of the same sign (Perron-Frobenius). Dividing by the sum both fixes the sign (negative/negative = positive) and normalizes to 1 in one step. No clipping needed.

Note: the lesson's own `docs/en.md` shows this same function in its correct form — only the shipped `.py` has the clip bug. So this fix also brings the code back in line with the docs.

## Test evidence (after fix)

- Analytical stationary: Sunny 0.5455, Rainy 0.1818, Cloudy 0.2727 — matches the empirical run (0.5469 / 0.1826 / 0.2705).
- Convergence check now behaves: max error 0.0531 → 0.0100 → 0.0104 → 0.0022 as chain length grows.
- Independent check with a different 3x3 transition matrix: pi @ P == pi, components non-negative, sums to 1.
- Full script runs to completion, exit 0.

## Draft PR title (rewrite as you like)

fix: stationary_distribution returns zeros for sign-flipped eigenvector (phase 1, lesson 22)

## Draft PR description (rewrite in your voice)

Running `phases/01-math-foundations/22-stochastic-processes/code/stochastic.py` prints an all-zero stationary distribution and a convergence table stuck at ~0.55 error. Cause: numpy returns the eigenvalue-1 eigenvector with negative sign, and the `np.clip(..., 0, None)` line zeroes it out before normalization. Replaced the clip-and-guard block with a single sum-normalization, which handles the sign and normalizes in one step (Perron-Frobenius guarantees same-sign components for a stochastic matrix). Matches the correct version already shown in the lesson's docs. Verified: analytical now agrees with the empirical distribution and the convergence check shrinks with chain length.

## Submit it (3 steps)

1. In your fork: `git checkout -b fix-lesson22-stationary-distribution`
2. Copy `my-builds/contrib/pr-01-fix-stationary-distribution/stochastic.py` over `phases/01-math-foundations/22-stochastic-processes/code/stochastic.py`, then commit and push.
3. Open the PR against `rohitg00/ai-engineering-from-scratch` with your rewritten title and description.

One fix per PR, per their contributing guide. This PR touches only the one `.py` file — no README/ROADMAP parser concerns, no need to run `site/build.js`.
