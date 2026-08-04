# 03 — Matrix Transformations

> A matrix is a machine that moves space; eigenvectors are the directions it cannot turn.

**Project:** From-scratch 2D transformation demos (rotate, scale, shear) proving order matters, power iteration to find the dominant eigenvector, numpy eigendecomposition of [[2,1],[1,2]] with full verification, and a demo of why RNN weights explode or vanish under repeated multiplication.

## What I built
- Rotation, scaling, and shear matrices applied to 2D points
- Proof that R@S != S@R (transformation order changes the result)
- Power iteration from scratch, matching numpy's dominant eigenvalue/eigenvector
- Eigendecomposition of [[2,1],[1,2]], verified with A == V @ D @ V^-1
- RNN demo: hidden state norms explode when eigenvalues > 1 and vanish when < 1

## Main points learned
- A matrix moves every point in space at once: rotate, stretch, or slant.
- Matrix order matters. Scale-then-rotate lands somewhere different than rotate-then-scale.
- An eigenvector is a direction the matrix does not turn. It only stretches it.
- The eigenvalue is the stretch factor for that direction.
- Eigendecomposition rewrites a matrix as: rotate into special axes, stretch, rotate back.
- Multiplying by the same matrix many times raises its eigenvalues to a power. Above 1 blows up, below 1 dies out.
- That power effect is exactly why plain RNNs struggle with long sequences.

## The algorithms, explained simply
**2D transformations.** Each matrix is a recipe for where the two axis arrows land; every other point follows along. Chaining matrices is like giving directions: "turn left then walk two blocks" ends somewhere different than "walk two blocks then turn left."

**Power iteration.** Hit a random vector with the matrix over and over, shrinking it back to length 1 each time. The strongest stretch direction wins a little more every round, like the loudest voice in a room echoing until it is the only one you hear.

**Eigendecomposition.** Take the matrix apart into three steps: line up with its special directions, stretch each one by its own factor, then line back up. Like rotating a photo so the stretch happens cleanly along the grid, then rotating it back.

**Repeated multiplication (RNN demo).** Applying the same matrix N times multiplies each eigen-direction by its eigenvalue N times, like compound interest. A rate above 1 snowballs; a rate below 1 melts away to nothing.

## How this shows up in AI
- Plain RNNs suffer exploding/vanishing gradients because the same weight matrix is applied at every time step; LSTMs and gradient clipping exist to fight this.
- Careful weight initialization keeps eigenvalue-like scale factors near 1 so signals survive deep networks.
- PCA finds the eigenvectors of the data's covariance: the directions where the data varies most.
- Power iteration is the core of PageRank and of quick top-component estimates on huge matrices.

## Run it
```
python3 project.py
```
