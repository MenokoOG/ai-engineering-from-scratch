# Course Feedback — 16-sampling-methods

**Reviewed:** code/sampling.py (ran, exit 0, PNG saved), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, Build It Step 10 (Gumbel-Softmax snippet).** The snippet takes a parameter named `logits` but computes `math.log(p) for p in logits`. If the inputs really are logits, taking their log is wrong (and crashes for logits <= 0); if they are probabilities, the name is wrong. The actual code file gets this right (`gumbel_softmax_sample(log_probs, ...)`). Fix the doc snippet to take `log_probs` (or `probs` and keep the log).

2. **docs/en.md, rejection sampling section: "Acceptance rate = 1/M".** Only true when the target pdf is normalized. The lesson's own truncated-normal example uses an unnormalized target (normal pdf restricted to [a, b], not renormalized), where the acceptance rate is Z/M (Z = mass of the target under the proposal range). Verified: the demo's M gives 1/M ≈ 0.84 but the observed (and theoretically correct) acceptance is ≈ 0.68. Add "for normalized p(x)" or state acceptance = Z/M in general.

## Nitpicks & suggestions

- `sample_exponential_inverse_cdf` calls `math.log(random.random())` — `random.random()` can return exactly 0.0, which raises ValueError. Rare, but `gumbel_sample` in the same file guards against it; be consistent. The Metropolis-Hastings snippet in the docs has the same unguarded `math.log(random.random())` (the code file guards it with `+ 1e-300`).
- The Gibbs demo prints "empirical correlation" but computes covariance (E[xy] - E[x]E[y]) without dividing by the standard deviations. Numbers look right only because the marginal variances are 1 by construction.
- `truncated_normal_demo` computes `norm_const` (lines 60-63) and never uses it — dead code.
- The bimodal MH demo prints means of 0.97/0.79/0.67 against a true mean of 0.6 without comment; a line noting that small proposal_std mixes poorly between modes would turn the discrepancy into the lesson it is meant to be.

## What's solid

- Every sampler is correct: I verified the inverse-CDF exponential means, the truncated-normal mean (0.2266 vs theoretical 0.2296), importance-sampling E[X^2] = 5.0, Gumbel-Max empirical frequencies vs true probs, the Gibbs conditional N(rho*y, 1-rho^2), and the top-p renormalization by hand — all match.
- The LLM-facing material (temperature, top-k, top-p, reparameterization trick, Gumbel-Softmax, straight-through) is accurate and well connected to practice; the quiz answers are all correct.
