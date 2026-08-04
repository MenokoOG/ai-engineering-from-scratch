# Master Cheat Sheet — Phase 1: Math Foundations

> Every algorithm, pattern, and hard-won fact from all 22 lessons. One page to rule them all.

**How to use this:** scan the topic you need. Each entry is: what it is, in plain English, plus the one fact you must not forget. Deeper explanations live in each lesson's `LESSON.md`.

---

## 1. The Big Map — which math powers which AI

| Math | Where it shows up in AI |
|---|---|
| Dot product / cosine similarity | Embedding search, "how similar are two meanings" |
| Matrix multiply | Every neural network layer is `relu(W @ x + b)` |
| Low-rank factorization | LoRA fine-tuning (store d×r + r×d instead of d×d) |
| Eigenvalues | Why RNN gradients explode (>1) or vanish (<1) |
| Derivatives + chain rule | Backpropagation — the entire training loop |
| Softmax + cross-entropy | The output layer and loss of every classifier/LLM |
| KL divergence | Training objective (CE = entropy + KL), RLHF penalties |
| PCA / SVD | Compression, embeddings, recommender systems |
| Broadcasting + einsum | Attention: `'bhtd,bhsd->bhts'` is the score matrix |
| Float formats | bf16 vs fp16 — why training doesn't overflow |
| Sampling (temp, top-k, top-p) | Every LLM generation knob you've ever turned |
| MCMC / Langevin | Diffusion models are noisy gradient descent, reversed |
| Fourier / rotations | RoPE positional encoding, conv-as-multiplication |
| Graph Laplacian | GNNs, spectral clustering, community detection |
| Markov chains | Diffusion forward process, RL environments |

---

## 2. Core algorithms, one breath each

### Linear algebra (lessons 01–03, 11, 12, 17)

- **Dot product** — multiply matching slots, add up. Big = pointing the same way.
- **Cosine similarity** — dot product with lengths divided out. Direction only. The standard "meaning similarity" for embeddings. Guard against zero-length vectors.
- **Matrix multiply** — (m×n) @ (n×p) → (m×p). Inner dims must match. It applies right-to-left: `R @ S` means "do S first, then R." Order matters.
- **Rank / linear independence** — vectors are independent iff rank = count. Rank tells you how much real information a matrix carries. det = 0 means rows are dependent: the matrix squashes space flat and has no inverse.
- **Eigenvector** — a direction the matrix only stretches, never turns. Power iteration (multiply, renormalize, repeat) finds the dominant one. Repeated multiplication raises eigenvalues to the Nth power: |λ|>1 explodes, |λ|<1 vanishes — the RNN gradient story.
- **SVD** — any matrix = rotate · scale · rotate (U Σ Vᵀ). Works on rectangular matrices, always exists. Keep the top-k singular values → best rank-k compression. A rank-4 copy of a 32×32 image kept 99.5% of its energy.
- **PCA** — center the data, take covariance, grab the top eigenvectors: the directions of most variance. sklearn computes it via SVD because forming XᵀX squares the condition number.
- **LoRA pattern** — a rank-r update to a d×d weight stores as A(d×r) @ B(r×d). Tiny storage, near-perfect reconstruction when the true change is low-rank.
- **Gaussian elimination + partial pivoting** — always divide by the biggest available pivot; tiny pivots amplify rounding into garbage.
- **LU decomposition** — factor once O(n³), solve each new right-hand side O(n²). The factor-once/solve-many pattern.
- **Cholesky** — LU's cheaper cousin for symmetric positive-definite matrices. The right tool for ridge regression `(XᵀX + λI)w = Xᵀy`.
- **Least squares** — no exact solution? Get the closest one. Normal equations are fine for demos, but they square the condition number — use QR/SVD when it matters.
- **Broadcasting rule** — right-align the shapes; each dim must match or be 1. `(8,1,6,1)` with `(7,1,5)` → `(8,7,6,5)`.
- **Einsum** — an index that appears in inputs but not the output gets summed away. `'bhtd,bhsd->bhts'` sums over d = attention scores.

### Calculus & optimization (lessons 04, 05, 08, 18)

- **Derivative** — the slope right here. **Gradient** — the uphill arrow for many variables. Training steps opposite it: `w = w − lr · dL/dw`. That single line IS deep learning.
- **Central difference** — `(f(x+h) − f(x−h)) / 2h` with h ≈ 1e-5. Use it to check analytic gradients: relative error below ~1e-7 means you're right.
- **Chain rule / backprop** — each op only knows its own local slope; backward() multiplies them in reverse topological order. Gradients must ACCUMULATE (`+=`): a value used twice gets gradient from both paths. Zero grads every step.
- **Reverse-mode autodiff wins** because one backward pass gives gradients for ALL parameters from one scalar loss.
- **GD stability limit** — lr < 2/curvature. Too big diverges abruptly, not gradually.
- **Momentum** — a rolling ball that remembers direction. **RMSProp** — per-parameter step sizing. **Adam** — both at once, plus bias correction. On easy bowls plain GD wins; on curved valleys (Rosenbrock) Adam/momentum crush it.
- **Cosine annealing** — start big, end tiny, smooth the ride down.
- **Convex** — the chord never dips below the curve; any local min is global; downhill always works. Deep nets aren't convex — SGD works anyway because high-dim bad points are mostly saddles and there are many good-enough minima.
- **Newton's method** — slope ÷ curvature: one perfect step on a quadratic. Unusable at LLM scale (Hessian is O(params²)).
- **KKT complementary slackness** — you only pay (λ > 0) for a wall you're leaning on; untouched constraints cost λ = 0.

### Probability & statistics (lessons 06, 07, 15, 16)

- **PMF vs PDF** — dice get probabilities; continuous curves get densities. Density can exceed 1; only areas are probabilities.
- **Central Limit Theorem** — averages of anything become bell-shaped. Why gaussians are everywhere.
- **Stable softmax** — subtract max(logits) before exp. Same answer, never overflows.
- **Cross-entropy = −log(p_true)** — "how surprised was the model by the right answer."
- **Log probabilities** — a thousand 0.1s multiplied underflows to exactly 0.0; their logs just add to −2302.6. LLMs live in log space.
- **Bayes' theorem** — belief update: posterior ∝ likelihood × prior. Rare-disease trap: 1/10,000 prevalence + 99% accurate test → still only ~1% chance you're sick, because false positives swamp true ones.
- **Laplace smoothing** — add fake counts so an unseen word can't nuke a whole class to log(0) = −inf.
- **MLE vs MAP** — MLE trusts only data; MAP adds a prior that acts like imaginary flips and washes out as data grows.
- **Beta updating is counting** — Beta(1,1) + 7 heads/3 tails = Beta(8,4). Order doesn't matter.
- **Bessel's n−1** — the sample mean sits closer to its own data than the truth does; dividing by n−1 fixes the shrinkage.
- **p-value** — the chance of seeing a gap this big if there were no real difference. "Significant" means "probably not zero," never "big." At n = 10M, a 0.03% gap is significant.
- **Multiple comparisons** — 20 tests at α = 0.05 → 64% chance of at least one false positive. Bonferroni: divide α by the test count.
- **Bootstrap** — resample your test set with replacement to get a confidence interval. For comparing models, resample the SAME indices for both (paired) or your CI lies.
- **Temperature** — divides logits before softmax. Changes gaps, never ranking. T<1 sharpens, T>1 flattens.
- **Top-k vs top-p** — top-k keeps a fixed count; top-p keeps the smallest set covering probability p, so it adapts to model confidence.
- **Reparameterization trick** — write z = μ + σ·ε so gradients flow through sampling (VAEs).
- **Metropolis-Hastings** — propose a move, accept by target ratio (normalizing constant cancels). Proposal too small = stuck in one mode; too big = frozen chain. Watch the acceptance rate.
- **Rejection sampling dies in high dimensions** — ball/cube volume ratio: 78% in 2D, 0.25% in 10D. That's why MCMC exists.

### Information theory (lesson 09)

- **Entropy** — average surprise. Fair die = high, loaded die = low.
- **CE(p,q) = H(p) + KL(p‖q)** — so minimizing cross-entropy = minimizing KL to the truth.
- **Perplexity = exp(cross-entropy)** — "perplexity 50" = as confused as choosing among 50 equally likely words.
- **Mutual information beats correlation** — y = x² has ~0 Pearson correlation but high MI. MI ≈ 0 only under true independence.

### Numerics (lessons 13, 14, 17, 19)

- **Machine epsilon** — float64 ~2.2e-16 (~16 digits), float32 ~1.2e-7 (~7 digits). Precision is relative: near 1e8, float32 neighbors are 8.0 apart. 0.1 + 0.2 ≠ 0.3 is normal.
- **Log-sum-exp** — same subtract-the-max trick as stable softmax; use it any time you sum exps.
- **Finite-difference U-curve** — h too big = truncation error; h too small = cancellation error; sweet spot ~1e-5.
- **bf16 vs fp16** — bf16 keeps float32's RANGE with only ~3 decimal digits. Range is what training needs (fp16 overflows at 65504) — that's why bf16 wins.
- **Condition number** — digits you can trust ≈ 16 − log₁₀(κ). At κ = 1e8 a hair's nudge in b swings x by amplification ~κ.
- **Norms** — L1 = city blocks, L2 = straight line, L∞ = worst coordinate.
- **L1 makes sparsity** — its pull is constant (sign), so weights snap to exact zero; L2's pull fades near zero, so weights shrink but survive. Lasso prunes, Ridge shrinks.
- **Mahalanobis** — distance measured in units of the data's own spread; the Euclidean-closest point can be the true outlier.
- **Wasserstein vs KL** — non-overlapping distributions: KL is infinite/useless; Wasserstein ("earth-mover") still gives a graded number. Why WGANs exist.
- **Complex multiply** — lengths multiply, angles add. e^(iθ) is a pure rotation. **RoPE**: rotate each q/k pair by position × θ; the attention score then depends only on the position GAP — relative position from absolute rotations. (Needs conj(k), not plain multiply.)
- **Roots of unity** — N evenly spaced points on the unit circle; they sum to exactly 0.

### Signals, graphs, randomness (lessons 20, 21, 22)

- **DFT / FFT** — decompose a signal into frequencies. DFT is O(n²); FFT splits even/odd for O(n log n) (~700× faster at n = 1024).
- **Convolution theorem** — convolution in time = multiplication in frequency.
- **Zero-padding** — interpolates the spectrum but adds NO true resolution. Frequency resolution = 1/recording-length.
- **Sinusoidal positional encoding** — sin/cos at geometric wavelengths; fast dims separate neighbors, slow dims separate far positions; every position unique.
- **BFS** — queue; finds shortest paths in unweighted graphs. **DFS** — stack; dives deep.
- **Graph Laplacian L = D − A** — number of zero eigenvalues = number of connected components.
- **Fiedler vector** — eigenvector of the 2nd-smallest Laplacian eigenvalue; its SIGNS cut the graph at its weakest bridge (spectral clustering).
- **GNN message passing** — new feature = nonlinearity(W @ mean of neighbors). That's it.
- **Markov property** — the future depends only on the present state, not the path here.
- **Stationary distribution** — where a Markov chain settles; find it by power iteration on the transition matrix, verify π·P = π.
- **Random walk** — expected distance grows as √n. 100× more steps = only 10× farther.
- **Langevin dynamics** — gradient descent + √(2T·dt) noise. T→0 collapses to the nearest minimum; higher T explores. The knob MCMC and diffusion samplers turn.
- **Diffusion forward process** — keep noising data until it's pure gaussian; SNR falls monotonically; the model learns to run it backwards, one denoise at a time.

---

## 3. Patterns worth stealing (they repeat everywhere)

1. **Subtract the max before exp** — softmax, log-sum-exp, anything exponential. Free stability.
2. **Work in log space** — products of probabilities become sums; nothing underflows.
3. **Factor once, solve many** — LU/Cholesky; same spirit as caching, precomputation, KV-cache.
4. **Check gradients numerically** — central difference vs analytic; relative error < 1e-7 = correct.
5. **Accumulate gradients with +=, zero them every step** — the two autodiff rules that bite hardest.
6. **From scratch first, library second** — write the loop version, assert it equals numpy/sklearn, then trust the library.
7. **Seed every RNG** — or your bug reports and demos are unreproducible.
8. **Compare with tolerance, never ==** — floats give you −0.0000 and 1e-16 residue everywhere. Eigenvector signs are arbitrary: compare with abs().
9. **Low-rank is the compression story** — SVD, PCA, LoRA, recommenders: same math, four costumes.
10. **Noise is a feature** — SGD minibatch noise escapes saddles; Langevin noise explores; diffusion noise trains. Don't reflexively kill randomness.

---

## 4. Numbers to memorize

| Fact | Value |
|---|---|
| float64 / float32 precision | ~16 digits / ~7 digits |
| fp16 max value | 65,504 (overflow risk) |
| Best finite-difference h | ~1e-5 |
| Gradient-check pass | relative error < 1e-7 |
| GD stability | lr < 2 / curvature |
| Digits lost to conditioning | ≈ log₁₀(κ) |
| (8,1,6,1) ⊕ (7,1,5) broadcasts to | (8,7,6,5) |
| 20 tests at α=0.05, false-positive chance | ~64% |
| Rare-disease posterior (1/10⁴ prior, 99% test) | ~1% |
| Uniform-over-50-words perplexity | exactly 50 |
| Random-walk distance after n steps | ~√n |

---

## 5. Gotchas that actually bit us

- A "random" 3×3 example matrix turned out singular (det = 0) and silently poisoned a different demo. Check your example data.
- Assigning instead of accumulating gradients (`=` not `+=`) gave 3 instead of 6 for y = x·x — wrong with no crash.
- lr = 0.5 un-converged a 21-parameter net; 0.2 trained it. Even toys are lr-sensitive.
- Paired bootstrap: resampling models independently instead of on shared indices silently doubles your CI width.
- numpy's reshape on a transpose silently copies; torch's `.view()` loudly fails. Same cause: strides. Use `.contiguous()` / `.reshape()`.
- `statistics.py` as a filename shadows Python's stdlib module and breaks imports (found in the course itself — lesson 15).
- Naming traps: PyTorch images are NCHW, TensorFlow NHWC — mixing them scrambles channels without erroring.
- MAP uses the Beta MODE, not the mean. Off-by-one in the formula gives quietly wrong answers.
- t-SNE is for looking, not preprocessing. Never feed it to a classifier.

---

## 6. Course bugs we found and reported (see each lesson's FEEDBACK.md)

- **22-stochastic-processes** — `stationary_distribution` in the shipped code clips an all-negative eigenvector to zeros and returns [0,0,0]; the doc's version is correct. Real bug.
- **15-statistics-for-ml** — `statistics.py` filename shadows the stdlib; breaks `import seaborn` from that directory.
- **13-numerical-stability** — several doc claims false in float64 (nan-max, softmax failure ranges), FP8 E4M3 max is 448 not 240, garbled leftover text.
- **17-linear-systems** — worked least-squares example prints [1.5, 1.7]; correct is [1.5, 1.6].
- **10-dimensionality-reduction** — kernel-PCA scaling stated inverted in docs (code is right).
- **08/09/11/12/14/16/20/21** — smaller doc/demo mismatches: demos that don't demonstrate their own claim, snippets calling the slow/wrong helper, promised output files never written.
- Quiz answer keys: correct in all 22 lessons.

---

*Built by Lawrence (with Rune Onyx) — classHuman AI. LAHA — Love All Humans Always.*
