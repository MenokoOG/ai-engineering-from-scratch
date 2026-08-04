# Course Feedback — 22-stochastic-processes

**Reviewed:** code/stochastic.py (ran with python3, exit 0 but wrong output), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Bugs found

## Bugs & errors

1. **code/stochastic.py, `MarkovChain.stationary_distribution` (lines 46-54): returns all zeros.** The eigenvector numpy returns for eigenvalue 1 can come out with all-negative entries (it does for the weather chain: [-0.857, -0.286, -0.429]). The code then does `np.clip(stationary, 0, None)`, which zeroes every entry; the sum is 0, so normalization is skipped and the function returns [0, 0, 0]. The demo visibly prints "Stationary distribution (analytical): Sunny 0.0000, Rainy 0.0000, Cloudy 0.0000". The correct answer is [6/11, 2/11, 3/11] ≈ [0.5455, 0.1818, 0.2727] (I verified pi·P = pi, and it matches the empirical run: 0.5469/0.1826/0.2705). Fix: normalize first, then clip/abs — e.g. `stationary = stationary / stationary.sum()` before any clipping (dividing by the negative sum flips the signs), as the doc's own snippet does. Note the docs/en.md Step 2 version of this function is written differently and works; only the shipped .py is broken.

2. **code/stochastic.py, `demo_markov_chain` convergence check (lines 154-159): meaningless output caused by bug 1.** The "convergence check" compares the empirical distribution to the broken all-zeros pi, so it prints "max error = 0.55" at every chain length and never converges. With the fixed pi, the error shrinks as expected (about 0.03 at 100 steps, 0.005 at 100000). Fixing bug 1 fixes this too, but the table as shipped actively teaches the wrong lesson (it looks like Markov chains don't converge).

## Nitpicks & suggestions

1. docs/en.md, weather example: "the stationary distribution might be [0.53, 0.18, 0.29]". The exact answer for the given matrix is [0.545, 0.182, 0.273]; since it is computable exactly, the doc may as well state the true values.
2. docs/en.md, "Ship It": promises `outputs/prompt-stochastic-process-advisor.md`, but stochastic.py writes no output file (no `open`/`write` calls). Same doc-code mismatch as lesson 21.
3. docs/en.md, mixing time: "spectral gap of P (1 minus the second-largest eigenvalue)" — should be second-largest eigenvalue in absolute value (matters for chains with negative eigenvalues).
4. demo_langevin: sampled variance is 2.09 vs the "expected 2.0" label. This is the known discretization bias of unadjusted Langevin at dt = 0.1, not a bug, but a one-line comment would preempt confusion.

## What's solid

- The samplers are textbook-correct: Langevin recovers N(3, 2) well, Metropolis-Hastings on the bimodal target gets a 50.2/49.8 mode split with correct per-mode means, and the random-walk demo confirms std = sqrt(n) (31.63 vs 31.62 over 10000 walks).
- The DDPM forward-process demo and the diffusion/Brownian-motion narrative in the docs match the actual DDPM formulation (x_t = sqrt(1-beta_t)·x_{t-1} + sqrt(beta_t)·noise).
