# Course Feedback — 07 Bayes' Theorem

**Reviewed:** code/bayes.py (ran with python3), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Clean

## Bugs & errors

None found.

- bayes.py runs end-to-end with no errors or warnings. All printed numbers check out: medical test posterior 0.0098, sequential second test 0.495, spam filter 0.9554, MAP estimates 0.6667 and 0.5714, Beta updates Beta(22, 20), A/B test P(B > A) about 0.92.
- Every worked example in docs/en.md was recomputed by hand or by script. All correct.
- The Marsaglia-Tsang gamma sampler used for Beta sampling is implemented correctly.
- All 5 quiz answers are marked correctly and the explanations are accurate.

## Nitpicks & suggestions

1. The `bayes()` function is specific to a binary test (sensitivity + false positive rate), not general Bayes. A comment saying so would help.
2. Docs say Bayesian A/B early stopping is "Safe at any point". The table hedges this, but it is a debated claim. Fine as written, just know it depends on the prior and model being right.
3. "99% accurate" is used to mean both sensitivity and specificity being 99%. The docs do define both, but calling it out explicitly ("accuracy here means both rates") would prevent confusion.
4. Exercise 2 asks what happens with smoothing=0. The code will raise a math domain error on log(0). That is presumably the point, but the exercise could say "expect a crash".

## What's solid

- The base-rate fallacy demo, sequential testing, conjugate Beta updates, and Bayesian A/B testing form a coherent story, and sequential-vs-batch equivalence is verified in code.
- The from-scratch Naive Bayes matches the docs exactly, uses log-space correctly, and the sklearn comparison is a nice touch.
