# Course Feedback — 14-norms-and-distances

**Reviewed:** code/distances.py (ran, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **code/distances.py, `demo_different_neighbors` (~line 429).** The section is titled "SAME DATA, DIFFERENT METRICS, DIFFERENT NEAREST NEIGHBORS" and the docs (Build It, Step 2) promise "the nearest neighbor changes depending on the distance metric." With the hardcoded seed (123) and query, all four metrics pick Point 0, and the "The metrics DISAGREE" message is behind an `if not all_same:` guard that never fires. The demo never demonstrates its headline claim. Fix: pick a query/dataset where the metrics actually disagree (easy to construct: put one point close in angle but far in magnitude).

2. **code/distances.py, `demo_knn_classification` (~line 566).** Same problem: titled "DISTANCE METRIC CHANGES THE PREDICTION," but all four metrics return the same 3 neighbors and predict class C. The query (2.8, 2.8) sits inside the C cluster, so nothing changes. Fix: place the query near a class boundary so at least one metric flips the vote.

3. **code/distances.py, `demo_regularization` (~line 613).** The demo runs penalty-only gradient descent (no data loss) for 50 steps at lr=0.1. The L1 step budget (50 × 0.1 = 5.0) exceeds the largest |weight| (2.995), so L1 zeroes out ALL weights, not just "small" ones, and the L2 weights shrink so far that the printed `L2 norm: 0.0000` and weights `[-0.0, 0.0, ...]` look like zeros too. The printed conclusions ("L1 drives 'small' weights to exactly zero", "L2 shrinks all weights but none reach exactly zero") are technically shown by the `Zeros: 0/10` count, but the display undercuts the story. Fix: fewer steps or smaller lr so L1 zeroes small weights while keeping the two large ones, and print L2 weights with more precision (e.g. scientific notation).

## Nitpicks & suggestions

- `random.seed(99)` in `demo_knn_classification` is set but never used (the demo is deterministic).
- `cosine_similarity` returns 0.0 for a zero vector; fine as a convention, but worth a comment since cosine is mathematically undefined there.
- `wasserstein_1d` assumes unit spacing between bins; a one-line note would prevent misuse with non-uniform bins.
- Docs "python -> pytorch: distance = 4" and the kitten/sitting DP table are correct (I verified both), as are the Wasserstein hand-examples — nice.

## What's solid

- Every distance function is mathematically correct: I verified L1/L2/Lp/L-inf, cosine, Mahalanobis (with the from-scratch Gauss-Jordan inverse), Jaccard, Levenshtein, KL, and the CDF-based 1D Wasserstein against hand computations.
- The docs are unusually good: the metric-selection table, the L1-diamond/L2-circle sparsity explanation, and the ANN/HNSW overview are accurate and practical. Quiz answers are all correct.
