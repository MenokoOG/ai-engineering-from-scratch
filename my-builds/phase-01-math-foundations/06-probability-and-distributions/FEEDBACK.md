# Course Feedback — 06 Probability and Distributions

**Reviewed:** code/probability.py (run, exit 0; matplotlib figure saved), docs/en.md, quiz.json (no Julia files in this lesson's code folder)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Log Probabilities" section.** Claims the product `0.01 * 0.003 * 0.02 * ...` hits "0.0 (underflow after ~30 terms)". Wrong for float64: 30 terms of that size is ~1e-72, and float64 holds down to ~1e-308 (underflow near ~154 factors of 0.01). I verified: `0.01**154` is still representable, `0.01**160` is subnormal. The quiz explanation for Q5 repeats the same claim ("quickly underflows to 0.0 with float64"). The log-prob lesson itself is right; the term count is off by ~5x. Fix: say "a few hundred terms" or cite float32, where ~19 factors of 0.01 do underflow.
2. **code/probability.py, Bernoulli sampling demo (line ~167).** With `random.seed(42)` and n=20, the demo prints "Empirical mean: 0.5000 (expected 0.3)". The sampler is correct — this is a ~4.8% unlucky draw baked in by the fixed seed — but as a printed teaching output it actively undermines the point. Fix: use n=1000 like the categorical demo, or pick a seed whose sample mean lands near 0.3.

## Nitpicks & suggestions

- `sample_normal_box_muller` calls `math.log(u1)` where `random.random()` can return exactly 0.0 → crash. Astronomically rare, but the standard fix (`1 - random.random()`) is one character of thought.
- `factorial`/`combinations` are defined but `combinations` is never used in the demo.
- Docs Step 7 shows placeholder pseudocode (`for x, mu, sigma in ...`) that won't run as written; it does say the real code is in probability.py, but a runnable mini-snippet would be better.

## What's solid

- All PMF/PDF formulas, expected value/variance (die: 3.5, 2.9167), conditional probability (1/3), softmax stability trick, log-sum-exp, joint/marginal table, and CLT demo are correct and verified at runtime.
- All five quiz correct answers are marked correctly; the softmax-trick and cross-entropy explanations are particularly good.
