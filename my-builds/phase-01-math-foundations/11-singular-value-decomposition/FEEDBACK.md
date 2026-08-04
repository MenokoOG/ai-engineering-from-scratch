# Course Feedback — 11 Singular Value Decomposition

**Reviewed:** code/svd.py (ran with python3), docs/en.md, quiz.json. Docs list "Languages: Python, Julia" and tell you to run `julia svd.jl`, but no Julia file exists in the lesson folder — nothing to review there.
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Use It" section.** References `code/svd.jl` ("The Julia version in code/svd.jl...") and gives a `julia svd.jl` run command, but the file is not shipped. The lesson folder contains only svd.py, docs, and quiz. Fix: ship the file or drop the Julia references.

2. **docs/en.md, "Build It" Step 3.** The snippet contains `image = np.random.seed(42)` — this assigns None to `image` (np.random.seed returns nothing). It happens to work only because `image` is reassigned two lines later, but it teaches a wrong pattern. Also, the "image" is `np.random.randn(200, 300)` — pure Gaussian noise has a flat singular-value spectrum, so the compression demo shows large errors at every k and undermines the section's own point ("natural images have rapidly decaying singular values"). The code file uses a proper structured synthetic image. Fix: `np.random.seed(42)` on its own line, and use a structured image (e.g., the sin/cos image from svd.py).

## Nitpicks & suggestions

1. Docs Step 1 `power_iteration` has no zero-vector guard; if the residual matrix is (near) zero before k iterations complete, `Mv / norm(Mv)` divides by zero and produces NaNs. The code file (svd.py) added the guard — the docs snippet did not pick it up.
2. Docs Step 5 pseudoinverse snippet inverts all singular values with no tolerance (`1.0 / S`). Fine for the full-rank example shown, but it will blow up on the singular-matrix case covered later. The code file's `pseudoinverse_via_svd` does it right with a tolerance.
3. In `demo_lsa`, term counts use substring matching (`doc.count(term)`), so "meow" matches "meows", but also e.g. "cat" would match "catch". Harmless with this vocabulary; fragile if a student edits the docs list.
4. The image-compression table happily shows k=200 with storage ratio 156.6% (bigger than the original). That is a good teaching moment — one sentence pointing it out would land it.

## What's solid

- The power-iteration + deflation SVD is correct: singular values match NumPy to 4+ decimals and reconstruction error is ~1e-10; A v_i = sigma_i u_i is verified in the output.
- Excellent breadth done correctly: Eckart-Young errors match theory exactly (rank-4 Frobenius error = sigma_5 = 3.0), the pseudoinverse min-norm and least-squares cases agree with lstsq, the condition-number-squaring demo is numerically real, and PCA-via-SVD matches sklearn. Quiz answers are all correct.
