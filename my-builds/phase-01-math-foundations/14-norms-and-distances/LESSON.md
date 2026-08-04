# 14 — Norms and Distances

> "How far apart?" has many right answers; pick the ruler that matches the question.

**Project:** L1/L2/Linf norms, cosine similarity, Mahalanobis distance, and 1D Wasserstein built from scratch, plus a tiny regression that shows L1 regularization zeroing out useless weights while L2 only shrinks them. A TypeScript version reuses the toy embeddings for a mini semantic search.

## What I built
- L1, L2, and Linf norms from scratch, asserted against numpy
- Manhattan vs Euclidean walk on a city grid (7 blocks vs 5 as-the-crow-flies)
- Cosine similarity on toy embeddings, showing it beats raw distance for meaning
- L1 vs L2 regularized regression trained with subgradient descent (7/10 weights zeroed by L1, 1/10 by L2)
- Mahalanobis distance on correlated 2D data: the Euclidean-closer point is the real outlier
- 1D Wasserstein vs KL on non-overlapping distributions: KL says infinity, Wasserstein says 2 vs 8
- project.ts: L1/L2/cosine plus a tiny nearest-neighbor search over toy documents

## Main points learned
- A norm is a ruler for one vector's size; a distance is that ruler applied to the gap between two.
- L1 adds up the blocks you walk; L2 flies straight; Linf only cares about the worst coordinate.
- Cosine similarity drops vector length and compares direction only — a long essay about cats still points the "cat" way.
- L1 regularization pulls every weight with constant force, so useless weights hit exactly zero. L2's pull fades near zero, so weights shrink but survive.
- Mahalanobis distance measures "how unusual is this point given how the data spreads," not just raw distance.
- KL divergence breaks (goes infinite) when distributions do not overlap; Wasserstein still gives a graded, useful number.
- Sparsity from L1 is feature selection for free: the surviving weights tell you what matters.

## The algorithms, explained simply
**L1 / L2 / Linf norms.** Three ways to score a trip: total blocks walked (L1), straight-line distance (L2), or just the longest single leg (Linf). Same trip, three different totals.

**Cosine similarity.** Divide the dot product by both lengths so only the angle remains. Like comparing two people's grocery carts by proportions instead of totals — a family-size cart and a single-person cart with the same mix count as identical taste.

**Subgradient descent with L1.** Regular gradient descent, but the L1 penalty adds a constant tug toward zero on every weight, like a flat monthly fee on each feature. Features that do not earn their fee get canceled to zero. L2's fee is proportional instead, so cheap weights are barely charged and never quite leave.

**Mahalanobis distance.** First learn how the data naturally stretches (its covariance), then measure distance in stretched units. Walking 2 miles along a highway is normal; 2 miles straight into the woods is weird — same mileage, different surprise.

**1D Wasserstein.** Treat each distribution as piles of dirt and ask how much work it takes to shovel one into the shape of the other. Two piles that do not touch are not "infinitely different" — moving dirt 2 bins is simply easier than moving it 8.

## How this shows up in AI
- Embedding search, RAG, and recommendation all rank by cosine similarity because meaning lives in direction, not vector length.
- Lasso (L1) gives sparse, interpretable models; weight decay in deep learning is the L2 side of this same demo.
- Mahalanobis-style whitened distances power anomaly detection and appear inside Gaussian classifiers.
- Wasserstein loss (WGAN) exists precisely because KL/JS give useless signals when generated and real data barely overlap.

## Run it
```
cd 14-norms-and-distances
python3 project.py
npx tsx project.ts
```
