"""Lesson 10 - Dimensionality Reduction: PCA from scratch + curse of dims + t-SNE notes."""
import numpy as np
from sklearn.decomposition import PCA as SkPCA

rng = np.random.default_rng(10)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def pca_from_scratch(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :k]           # (d, k)
    projected = Xc @ components           # (n, k)
    evr = eigvals / eigvals.sum()
    return projected, components, eigvals, evr, mean


# ---------------------------------------------------------------
section("1) PCA FROM SCRATCH: center -> covariance -> eigendecompose -> project")
# 3D data that mostly lives on a tilted 2D plane
n = 500
latent = rng.normal(size=(n, 2)) * np.array([3.0, 1.0])
mix = np.array([[1.0, 0.2, 0.5], [0.3, 1.0, -0.4]])
X = latent @ mix + rng.normal(scale=0.1, size=(n, 3)) + np.array([5.0, -2.0, 1.0])

proj, comps, eigvals, evr, mean = pca_from_scratch(X, k=2)
print(f"data shape {X.shape}, eigenvalues of covariance: {np.round(eigvals, 4)}")
print(f"explained-variance ratio: {np.round(evr, 4)}")
print(f"top-2 components capture {evr[:2].sum() * 100:.2f}% of variance")
assert evr[:2].sum() > 0.99

# ---------------------------------------------------------------
section("2) VERIFY AGAINST SKLEARN PCA")
sk = SkPCA(n_components=2).fit(X)
sk_proj = sk.transform(X)
# eigenvector signs are arbitrary; align sign per component before comparing
for j in range(2):
    if np.dot(comps[:, j], sk.components_[j]) < 0:
        comps[:, j] *= -1
        proj[:, j] *= -1
assert np.allclose(comps.T, sk.components_, atol=1e-8)
assert np.allclose(proj, sk_proj, atol=1e-8)
assert np.allclose(evr[:2], sk.explained_variance_ratio_, atol=1e-8)
print("components match sklearn:            OK (up to sign flip)")
print("projected coordinates match sklearn: OK")
print("explained-variance ratio matches:    OK")
print("PASS: from-scratch PCA == sklearn PCA")

# ---------------------------------------------------------------
section("3) CURSE OF DIMENSIONALITY: distances concentrate")
print("1000 random uniform points per dimension count; ratio -> 1 means")
print("'nearest' and 'farthest' neighbors become almost the same distance.")
print(f"{'dims':>6}{'min dist':>10}{'mean dist':>11}{'max dist':>10}{'min/max':>9}")
ratios = []
for d in [2, 10, 100, 1000]:
    pts = rng.uniform(size=(1000, d))
    ref = rng.uniform(size=(1, d))
    dists = np.linalg.norm(pts - ref, axis=1)
    ratio = dists.min() / dists.max()
    ratios.append(ratio)
    print(f"{d:>6}{dists.min():>10.3f}{dists.mean():>11.3f}{dists.max():>10.3f}{ratio:>9.3f}")
assert ratios == sorted(ratios), "ratio should grow toward 1 as dims grow"
assert ratios[-1] > 0.8
print("PASS: as dims grow, all points look equally far away")

# ---------------------------------------------------------------
section("4) RECONSTRUCT FROM k COMPONENTS AND MEASURE ERROR")
print(f"{'k':>3}{'variance kept':>15}{'reconstruction MSE':>20}")
errors = []
for k in [1, 2, 3]:
    p_k, c_k, _, evr_k, mu = pca_from_scratch(X, k)
    X_rec = p_k @ c_k.T + mu
    mse = float(np.mean((X - X_rec) ** 2))
    errors.append(mse)
    print(f"{k:>3}{evr_k[:k].sum() * 100:>14.2f}%{mse:>20.6f}")
assert errors == sorted(errors, reverse=True)
assert errors[-1] < 1e-20  # k = full dim -> perfect reconstruction
print("PASS: more components -> lower error; all components -> exact rebuild")

# ---------------------------------------------------------------
section("5) WHY t-SNE IS FOR VISUALIZATION ONLY")
# tiny neighbor-preservation idea: t-SNE only tries to keep each point's
# nearest neighbors close; it does not preserve global distances or axes.
a = np.array([0.0, 0.0])
b = np.array([1.0, 0.0])
c = np.array([50.0, 0.0])
print("Toy fact: t-SNE optimizes 'who is my neighbor', not 'how far apart'.")
print(f"  true dists: |a-b| = {np.linalg.norm(a-b):.0f}, |a-c| = {np.linalg.norm(a-c):.0f}")
print("  a valid t-SNE layout may show a-b close and a-c 'somewhere far',")
print("  but the 50-vs-1 ratio, cluster sizes, and axes mean nothing.")
print("Notes:")
print("  - t-SNE output is not a linear map: no .transform() for new points.")
print("  - distances between clusters in the plot are not meaningful.")
print("  - different random seeds give different pictures.")
print("  - so: never feed t-SNE coordinates to a classifier as features;")
print("    use PCA (a real, reusable linear projection) for preprocessing.")

print("\nALL CHECKS PASSED")
