# 10 — Dimensionality Reduction
> Most high-dimensional data secretly lives on a much smaller stage.

**Project:** Built PCA completely from scratch (center, covariance, eigendecomposition, project, explained-variance ratio) and matched sklearn exactly. Also demonstrated the curse of dimensionality, reconstruction error vs number of components, and why t-SNE is a picture tool, not a preprocessing tool.

## What I built
- PCA from scratch: center the data, build the covariance matrix, eigendecompose it, project onto top components
- Explained-variance ratio: how much of the data's spread each component captures
- Verification that components, projections, and variance ratios match sklearn's PCA (up to sign flips)
- Curse-of-dimensionality demo: nearest and farthest distances converge as dimensions grow
- Reconstruction from k components with error table (full k rebuilds the data exactly)
- Plain-English notes on why t-SNE is for visualization only

## Main points learned
- PCA finds the directions where the data spreads out most, in order of importance.
- Centering first matters: PCA measures spread around the mean, not around zero.
- Eigenvector signs are arbitrary. Your PCA and sklearn's can disagree by a minus sign and both be right.
- "95% variance with 50 components" means the data effectively lives in a 50-dimensional slice.
- In high dimensions, everything is roughly the same distance from everything. Nearest-neighbor logic degrades.
- Keeping more components always lowers reconstruction error; keeping all of them makes it zero.
- t-SNE keeps neighbors, not distances. Its plots are maps for the eye, not features for a model.

## The algorithms, explained simply
**PCA.** Find the single direction the data cloud is most stretched, then the next most-stretched direction at right angles to it, and so on. Like photographing a baguette: the best first camera angle captures its length, the second its width — after two angles you've basically seen the whole thing.

**Explained-variance ratio.** Each direction gets a score: what fraction of the total spread it accounts for. It's a budget report — "component 1 explains 90% of the action" tells you where the story is.

**Reconstruction.** Project down to k numbers, then multiply back up to the original space. Like summarizing a book in k sentences and rewriting it from the summary — more sentences, closer to the original.

**t-SNE (the idea).** For each point, note who its close friends are, then arrange points in 2D so friends stay together. Like a party seating chart: friends sit near friends, but the distance between two far-apart tables means nothing.

## How this shows up in AI
- Embedding spaces are heavily compressed views: a few hundred dimensions carry meaning for millions of words. PCA is the simplest version of that idea.
- The curse of dimensionality is why vector search needs good learned embeddings, not raw pixel or one-hot distances.
- LoRA fine-tuning bets that weight updates are low-rank — the same "few directions carry the signal" insight as PCA.
- t-SNE/UMAP plots of embeddings are for human inspection only; models should consume PCA or the raw embeddings.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/10-dimensionality-reduction
python3 project.py
```
