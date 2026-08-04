"""Lesson 11 - Singular Value Decomposition: geometry, compression, PCA link, recommendations."""
import numpy as np

rng = np.random.default_rng(11)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) SVD = ROTATE, SCALE, ROTATE (verified by reconstruction)")
A = np.array([[3.0, 1.0], [1.0, 2.0]])
U, s, Vt = np.linalg.svd(A)
print(f"A =\n{A}")
print(f"singular values: {np.round(s, 4)}")
print(f"U (rotation/reflection of outputs):\n{np.round(U, 4)}")
print(f"V^T (rotation/reflection of inputs):\n{np.round(Vt, 4)}")
assert np.allclose(U @ U.T, np.eye(2)), "U is orthogonal (pure rotation/reflection)"
assert np.allclose(Vt @ Vt.T, np.eye(2)), "V is orthogonal"
assert np.allclose(U @ np.diag(s) @ Vt, A), "U Sigma V^T rebuilds A exactly"
print("checks: U orthogonal OK, V orthogonal OK, U @ diag(s) @ V^T == A OK")

v = np.array([1.0, 1.0])
step1 = Vt @ v
step2 = np.diag(s) @ step1
step3 = U @ step2
print(f"apply to v={v}: rotate -> {np.round(step1,4)}, "
      f"scale -> {np.round(step2,4)}, rotate -> {np.round(step3,4)}")
assert np.allclose(step3, A @ v)
print("PASS: rotate-scale-rotate gives the same answer as A @ v")

# ---------------------------------------------------------------
section("2) LOW-RANK APPROXIMATION of a 32x32 'image'")
size = 32
yy, xx = np.mgrid[0:size, 0:size]
img = (np.sin(xx / 3.0) * np.cos(yy / 4.0)
       + 0.5 * ((xx // 8 + yy // 8) % 2)
       + 0.05 * rng.normal(size=(size, size)))
U2, s2, Vt2 = np.linalg.svd(img)
total_energy = float(np.sum(s2 ** 2))
print("keep only the top-k singular values, rebuild, measure error:")
print(f"{'rank k':>7}{'numbers stored':>15}{'rel. error':>12}{'energy kept':>13}")
prev_err = np.inf
for k in [1, 2, 4, 8, 16, 32]:
    approx = U2[:, :k] @ np.diag(s2[:k]) @ Vt2[:k, :]
    rel_err = float(np.linalg.norm(img - approx) / np.linalg.norm(img))
    stored = k * (size + size + 1)
    energy = 100 * float(np.sum(s2[:k] ** 2) / total_energy)
    print(f"{k:>7}{stored:>15}{rel_err:>12.4f}{energy:>12.2f}%")
    assert rel_err <= prev_err + 1e-12
    prev_err = rel_err
assert prev_err < 1e-10, "full rank rebuilds the image exactly"
print(f"(original stores {size*size} numbers; rank-8 stores {8*(2*size+1)})")
print("PASS: error drops as rank grows; full rank is exact")

# ---------------------------------------------------------------
section("3) PCA VIA SVD (this is what sklearn actually does)")
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5)) + rng.normal(size=5)
Xc = X - X.mean(axis=0)
n = X.shape[0]

# route 1: eigendecomposition of covariance
cov = Xc.T @ Xc / (n - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# route 2: SVD of centered data (no covariance matrix ever built)
U3, s3, Vt3 = np.linalg.svd(Xc, full_matrices=False)
eigvals_from_svd = s3 ** 2 / (n - 1)

for j in range(5):  # align arbitrary signs
    if np.dot(eigvecs[:, j], Vt3[j]) < 0:
        eigvecs[:, j] *= -1
assert np.allclose(eigvals, eigvals_from_svd, atol=1e-10)
assert np.allclose(eigvecs.T, Vt3, atol=1e-8)
print("eigendecomposition of covariance vs SVD of centered data:")
print(f"  variances match:  {np.round(eigvals, 4)}")
print(f"  components match: OK (rows of V^T == eigenvectors, up to sign)")
print("why sklearn uses SVD: skips forming X^T X (which squares the")
print("condition number, hurting precision) and is faster for tall matrices.")
print("PASS: singular values^2 / (n-1) == covariance eigenvalues")

# ---------------------------------------------------------------
section("4) MOVIE RATINGS: predict a missing rating with truncated SVD")
# rows = 6 users, cols = 5 movies; two taste groups (sci-fi fans, romance fans)
R = np.array([
    [5, 5, 4, 1, 1],
    [4, 5, 5, 2, 1],
    [5, 4, 5, 1, 2],
    [1, 2, 1, 5, 5],
    [2, 1, 1, 4, 5],
    [1, 1, 2, 5, 4],
], dtype=float)
held_user, held_movie = 2, 0
true_rating = R[held_user, held_movie]
R_obs = R.copy()
R_obs[held_user, held_movie] = np.nan

baseline = float(np.nanmean(R_obs[:, held_movie]))  # movie-average guess

# standard recipe: remove each user's average, then iterate
# fill missing with 0 -> truncated SVD -> use its value as new fill -> repeat
mask = np.isnan(R_obs)
user_means = np.nanmean(R_obs, axis=1, keepdims=True)
C = np.where(mask, 0.0, R_obs - user_means)
k = 2  # two taste factors
for _ in range(200):
    U4, s4, Vt4 = np.linalg.svd(C, full_matrices=False)
    C_hat = U4[:, :k] @ np.diag(s4[:k]) @ Vt4[:k, :]
    C[mask] = C_hat[mask]  # only overwrite the missing cell

pred = float(C[held_user, held_movie] + user_means[held_user, 0])
print(f"hidden: user {held_user} rating for movie {held_movie} (true = {true_rating:.0f})")
print(f"movie-average guess:          {baseline:.2f} (error {abs(baseline-true_rating):.2f})")
print(f"rank-{k} SVD prediction:        {pred:.2f} (error {abs(pred-true_rating):.2f})")
print(f"singular values: {np.round(s4, 2)} -> top 2 dominate = 2 taste groups")
print("(loop: subtract user means, fill hole, rank-2 SVD, refill, repeat)")
assert abs(pred - true_rating) < abs(baseline - true_rating)
assert abs(pred - true_rating) < 0.6
print("PASS: truncated SVD beats the mean by exploiting user-taste structure")

print("\nALL CHECKS PASSED")
