# Course Feedback — 10 Dimensionality Reduction

**Reviewed:** code/dim_reduction.py (ran with python3; downloads MNIST), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Kernel PCA" algorithm, step 4.** It says: "The top eigenvectors (scaled by 1/sqrt(eigenvalue)) are the projections." Wrong direction. For unit-norm eigenvectors a of the centered kernel matrix, the projections of the training points are a * sqrt(eigenvalue). Scaling by 1/sqrt(eigenvalue) gives the normalized coefficient vector alpha, not the projection. Verified numerically: the course's own `kernel_pca()` returns vecs * sqrt(lambda) and matches sklearn's KernelPCA to 1e-15; the docs' 1/sqrt scaling does not. Fix: change to "scaled by sqrt(eigenvalue)" (or explain the alpha-vs-projection distinction).

2. **code/dim_reduction.py, `demo_pca_preprocessing` (lines ~164-194) + docs "Use It".** LogisticRegression is fit on unscaled pixel data and fails to converge at every k (a wall of sklearn ConvergenceWarnings floods the output). Because of this, accuracy is not monotone: k=50 gives 0.900 but k=100 drops to 0.845. The docs then claim "Performance plateaus well before 784 dimensions" — the actual output shows a dip, not a plateau. Fix: scale the data (divide by 255 or StandardScaler) and/or raise max_iter; then the plateau story holds and the warnings disappear.

## Nitpicks & suggestions

1. `demo_sklearn_comparison` prints "Max absolute difference (sign-invariant): 0.0925". That looks alarming but is ~6e-5 relative to the ~1500-magnitude MNIST projections (eigh-vs-SVD float noise). Printing a relative difference would avoid scaring students.
2. Reconstruction-error table: the "Explained Var" column shows the ratio of the k-th (last) component only. The header does not say that; "Marginal var of PC k" would be clearer.
3. Docs claim "Reconstruction error = sum of eigenvalues NOT included". True for per-sample squared error; the code's MSE divides by the number of features too, so the printed numbers differ from dropped-eigenvalue sums by a factor of d. Worth one clarifying sentence.
4. Prerequisites line calls Lesson 03 "Eigenvalues & Eigenvectors", but Lesson 03 is "Matrix Transformations".
5. "Ship It" references `outputs/skill-dimensionality-reduction.md`, which does not exist in the lesson folder.

## What's solid

- The from-scratch PCA is correct (verified against sklearn: identical explained-variance ratios), and the kernel PCA implementation is genuinely right, including the subtle projection scaling and kernel centering.
- The concentric-circles kernel PCA demo and the reconstruction-error-vs-k table make the "effective dimensionality" idea concrete. Quiz answers are all correct.
