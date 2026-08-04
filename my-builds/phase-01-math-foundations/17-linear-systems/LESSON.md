# 17 — Linear Systems

> Most of applied math is "solve Ax = b" — the craft is doing it fast and without amplifying noise.

**Project:** A from-scratch linear-solver toolkit: Gaussian elimination with partial pivoting, LU factorization reused across many right-hand sides, least squares by normal equations, Cholesky-powered ridge regression, and a condition-number demo where a 1e-10 nudge to b swings the answer by 1%.

## What I built
- Gaussian elimination with partial pivoting, plus a no-pivot run that gets x[0] completely wrong
- LU decomposition (PA = LU) with forward/back substitution; 30 right-hand sides solved ~29x faster by reusing L and U
- Least squares via normal equations for a 50-point line fit (no exact solution, best fit found), matched against numpy lstsq
- Cholesky decomposition from scratch, used to solve the ridge system (X^T X + lambda I) w = X^T y
- Condition-number demo: kappa ~ 4e8 matrix where a tiny change in b is amplified ~kappa times in x

## Main points learned
- Elimination works by subtracting rows to create zeros, then back-substituting from the bottom up.
- Dividing by a tiny pivot multiplies rounding error into garbage. Pivoting = always divide by the biggest available number.
- LU splits the expensive work (factoring A) from the cheap work (two triangle solves per b). Factor once, solve forever.
- An overdetermined system has no exact answer; least squares finds the answer with the smallest total squared miss.
- Cholesky is LU's cheaper cousin for symmetric positive definite matrices — about half the work, guaranteed stable.
- The condition number kappa says how much A amplifies input noise; you lose about log10(kappa) digits of trust.
- Ridge's lambda I term does double duty: it regularizes the model and makes the matrix better conditioned.

## The algorithms, explained simply
**Gaussian elimination with partial pivoting.** Clean out one column at a time by subtracting multiples of a chosen row, then solve from the bottom up. Pivoting means always choosing the row with the biggest number in that column as your tool — like prying with the sturdiest crowbar available instead of a toothpick that snaps and ruins the job.

**LU decomposition.** Record the elimination steps themselves as a matrix L, keeping the cleaned-up result as U. It is like writing down a recipe while cooking: the first meal takes full effort, but every new right-hand side is just reheating — two quick triangle solves.

**Least squares via normal equations.** With more equations than unknowns, you cannot satisfy everyone, so multiply both sides by A-transpose to get a small solvable system whose answer minimizes total squared error. Like scheduling one meeting for fifty busy people: no time works for all, so you pick the slot with the least total pain.

**Cholesky decomposition.** For symmetric positive definite matrices, split A into L times L-transpose — one triangle that is its own mirror image. Since the matrix is symmetric, you only compute half of it, like folding paper in half before cutting a symmetric shape.

**Condition number.** Solve the system, then wiggle b slightly and watch how wildly x moves. A well-conditioned system is a firm doorknob; an ill-conditioned one is a hair trigger where a breath on the input slams the output.

## How this shows up in AI
- Ridge regression and Gaussian-process training solve exactly the (X^T X + lambda I) w = X^T y system, almost always via Cholesky.
- Least squares is the closed-form ancestor of every regression loss; linear probes on frozen embeddings still use it directly.
- Second-order and natural-gradient optimizers (and the linear solves inside them) live or die by conditioning; normalization layers exist partly to keep things well-conditioned.
- The factor-once-solve-many LU pattern is the same amortization mindset as caching KV pairs in transformer inference: pay the big cost once, reuse it cheaply.

## Run it
```
cd 17-linear-systems
python3 project.py
```
