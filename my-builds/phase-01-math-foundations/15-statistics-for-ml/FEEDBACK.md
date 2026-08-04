# Course Feedback — 15-statistics-for-ml

**Reviewed:** code/statistics.py (ran, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Bugs found

## Bugs & errors

1. **code/statistics.py — the filename itself.** `statistics.py` shadows Python's stdlib `statistics` module. Running the file directly works (it only imports `math` and `random`), but any other script, notebook, or library run from the `code/` directory that imports `statistics` — directly or transitively — breaks. Verified: with this file on sys.path, `import seaborn` fails with `ImportError: cannot import name 'NormalDist' from 'statistics' (.../statistics.py)`. A student who later does `from statistics import mean` expecting the stdlib (or adds pandas/seaborn plotting next to it) will hit confusing errors. Fix: rename to `stats_ml.py` or `ml_statistics.py`.

2. **code/statistics.py, `t_cdf_approx` / `_regularized_beta` (~line 149).** The midpoint-rule integration of the incomplete beta is inaccurate for small |t| because the integrand (1-t)^(-1/2) is near-singular at the upper limit. Verified against scipy: for t=0.1, df=49 the two-sided p-value comes out 0.8617 vs the true 0.9208 (off by ~0.06); t=0.05, df=9 gives 0.9398 vs 0.9612. Near the decision region it is fine (t=2.39, df=49: 0.0207 vs 0.0207), so significance calls in the demos are all correct, but the function is presented as a general p-value calculator. Fix: more steps near the singularity, a substitution, or a note that accuracy degrades for p near 1. (The chi-squared p-value has a smaller version of the same issue at df=1: 0.3281 vs scipy 0.3173.)

## Nitpicks & suggestions

- Sign conventions are inconsistent: `t_statistic_two_sample` computes (m1 - m2) while `cohens_d` computes (m2 - m1), so the demo prints t = -2.46 alongside d = +0.64 for the same comparison. Confusing for students; align them.
- In the "statistical vs practical significance" demo, the fixed seed makes the SMALL sample (n=30) also come out significant (p=0.028, d=0.58 "medium") for a true effect of 0.1. It accidentally demonstrates winner's curse rather than the intended contrast. A different seed or averaging over repeats would make the point cleanly.
- `mode()` silently returns the smallest of tied modes; a note about multimodal data would help.
- `bootstrap_statistic` uses `std_dev(..., sample=True)` for the SE; fine, but worth one comment line.

## What's solid

- The core statistics are all correct: I verified the t-tests, Welch df, chi-squared statistic, Pearson/Spearman, covariance matrix, percentile interpolation, and bootstrap CIs against scipy/known values; significance decisions in every demo match scipy.
- The docs are excellent and honest: correct p-value interpretation, Bonferroni math (1 - 0.95^20 = 0.64), CLT caveats, and a good "common mistakes in ML papers" list. Quiz answers are all correct.
