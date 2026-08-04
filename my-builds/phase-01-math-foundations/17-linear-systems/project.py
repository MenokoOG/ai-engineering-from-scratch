"""Lesson 17: Linear systems. Solving Ax = b well, fast, and without blowing up."""
import math
import time
import numpy as np

np.random.seed(17)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) Gaussian elimination: why partial pivoting matters")


def gauss_solve(A, b, pivot=True):
    A = [row[:] for row in A]
    b = list(b)
    n = len(A)
    for col in range(n):
        if pivot:
            best = max(range(col, n), key=lambda r: abs(A[r][col]))
            A[col], A[best] = A[best], A[col]
            b[col], b[best] = b[best], b[col]
        for r in range(col + 1, n):
            factor = A[r][col] / A[col][col]
            for c in range(col, n):
                A[r][c] -= factor * A[col][c]
            b[r] -= factor * b[col]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = sum(A[r][c] * x[c] for c in range(r + 1, n))
        x[r] = (b[r] - s) / A[r][r]
    return x


A = [[1e-17, 1.0], [1.0, 1.0]]
b = [1.0, 2.0]
x_np = np.linalg.solve(np.array(A), np.array(b))
x_nopiv = gauss_solve(A, b, pivot=False)
x_piv = gauss_solve(A, b, pivot=True)
print(f"A = [[1e-17, 1], [1, 1]], b = [1, 2], true x ~ {np.round(x_np, 6)}")
print(f"no pivoting  : x = {x_nopiv}   <- x[0] is garbage")
print(f"with pivoting: x = {[round(v, 6) for v in x_piv]}")
assert abs(x_nopiv[0] - x_np[0]) > 0.5          # catastrophically wrong
assert np.allclose(x_piv, x_np)
print("dividing by the tiny 1e-17 pivot amplified rounding error into nonsense")
print("swapping rows to divide by the biggest number available fixes it")

A2 = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
b2 = [8.0, -11.0, -3.0]
assert np.allclose(gauss_solve(A2, b2), np.linalg.solve(np.array(A2), np.array(b2)))
print("3x3 sanity check vs numpy: passed")

# ---------------------------------------------------------------
section("2) LU decomposition: factor once, solve many b's cheaply")


def lu_decompose(A):
    """PA = LU with partial pivoting. Returns L, U, perm (row order)."""
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    perm = list(range(n))
    for col in range(n):
        best = max(range(col, n), key=lambda r: abs(U[r][col]))
        U[col], U[best] = U[best], U[col]
        L[col], L[best] = L[best], L[col]
        perm[col], perm[best] = perm[best], perm[col]
        L[col][col] = 1.0
        for r in range(col + 1, n):
            factor = U[r][col] / U[col][col]
            L[r][col] = factor
            for c in range(col, n):
                U[r][c] -= factor * U[col][c]
    return L, U, perm


def forward_sub(L, b):
    n = len(L)
    y = [0.0] * n
    for r in range(n):
        y[r] = (b[r] - sum(L[r][c] * y[c] for c in range(r))) / L[r][r]
    return y


def back_sub(U, y):
    n = len(U)
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = sum(U[r][c] * x[c] for c in range(r + 1, n))
        x[r] = (y[r] - s) / U[r][r]
    return x


def lu_solve(L, U, perm, b):
    bp = [b[i] for i in perm]
    return back_sub(U, forward_sub(L, bp))


n = 60
A_big = np.random.randn(n, n) + n * np.eye(n)
L, U, perm = lu_decompose(A_big.tolist())
P = np.eye(n)[perm]
assert np.allclose(np.array(L) @ np.array(U), P @ A_big)
print(f"factored a {n}x{n} matrix: L @ U == P @ A verified")

many_bs = [np.random.randn(n).tolist() for _ in range(30)]
t0 = time.perf_counter()
xs_lu = [lu_solve(L, U, perm, bb) for bb in many_bs]
t_lu = time.perf_counter() - t0
t0 = time.perf_counter()
xs_full = [gauss_solve(A_big.tolist(), bb) for bb in many_bs]
t_full = time.perf_counter() - t0
for x_lu, bb in zip(xs_lu, many_bs):
    assert np.allclose(x_lu, np.linalg.solve(A_big, bb))
print(f"solving 30 right-hand sides:")
print(f"  full elimination each time : {t_full * 1000:7.1f} ms  (redoes the hard work)")
print(f"  reuse L,U + two triangles  : {t_lu * 1000:7.1f} ms  ({t_full / t_lu:.0f}x faster)")
assert t_lu < t_full
print("factor once (expensive), then each new b costs only two easy triangle solves")

# ---------------------------------------------------------------
section("3) least squares via normal equations (overdetermined)")

m = 50
t = np.linspace(0, 10, m)
true_slope, true_intercept = 2.5, -1.0
y_data = true_slope * t + true_intercept + 0.5 * np.random.randn(m)
X = np.column_stack([t, np.ones(m)])          # 50 equations, 2 unknowns
print(f"{m} noisy points, fitting y = slope*t + intercept: no exact solution exists")

AtA = (X.T @ X).tolist()
Atb = (X.T @ y_data).tolist()
w = gauss_solve(AtA, Atb)                     # solve the 2x2 normal equations
w_np, *_ = np.linalg.lstsq(X, y_data, rcond=None)
print(f"normal equations fit : slope = {w[0]:.4f}, intercept = {w[1]:.4f}")
print(f"numpy lstsq          : slope = {w_np[0]:.4f}, intercept = {w_np[1]:.4f}")
print(f"true generating line : slope = {true_slope}, intercept = {true_intercept}")
assert np.allclose(w, w_np)
residual = np.linalg.norm(X @ np.array(w) - y_data)
worse = np.linalg.norm(X @ (np.array(w) + 0.1) - y_data)
assert residual < worse
print("the fit minimizes total squared miss; nudging it any way makes the miss bigger")

# ---------------------------------------------------------------
section("4) Cholesky for SPD systems: ridge regression")


def cholesky(A):
    """A = L @ L.T for symmetric positive definite A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def cholesky_solve(A, b):
    L = cholesky(A)
    Lt = [list(col) for col in zip(*L)]
    return back_sub(Lt, forward_sub(L, b))


d = 8
Xr = np.random.randn(120, d)
w_ridge_true = np.random.randn(d)
yr = Xr @ w_ridge_true + 0.1 * np.random.randn(120)
lam = 0.5
G = Xr.T @ Xr + lam * np.eye(d)               # SPD by construction
rhs = Xr.T @ yr

L_chol = cholesky(G.tolist())
assert np.allclose(np.array(L_chol) @ np.array(L_chol).T, G)
w_ridge = cholesky_solve(G.tolist(), rhs.tolist())
assert np.allclose(w_ridge, np.linalg.solve(G, rhs))
print(f"ridge system (X^T X + {lam} I) w = X^T y, d = {d}")
print(f"my Cholesky solution matches numpy: {np.allclose(w_ridge, np.linalg.solve(G, rhs))}")
print(f"first 4 weights: {np.round(w_ridge[:4], 4)}")
print("Cholesky = half the work of LU, and it exploits the symmetry ridge guarantees")

# ---------------------------------------------------------------
section("5) condition number: how many digits can you trust?")

eps = 1e-8
A_ill = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
kappa = np.linalg.cond(A_ill)
print(f"A = [[1, 1], [1, 1 + 1e-8]],  condition number kappa = {kappa:.2e}")
print(f"float64 carries ~16 digits; expect to lose ~log10(kappa) = {math.log10(kappa):.0f} of them")

b0 = np.array([2.0, 2.0 + eps])
x0 = np.linalg.solve(A_ill, b0)               # exact answer is [1, 1]
b1 = b0 + np.array([0.0, 1e-10])              # nudge b in the 10th digit
x1 = np.linalg.solve(A_ill, b1)
rel_b = np.linalg.norm(b1 - b0) / np.linalg.norm(b0)
rel_x = np.linalg.norm(x1 - x0) / np.linalg.norm(x0)
print(f"x for b          = {x0}")
print(f"x for b + 1e-10  = {x1}")
print(f"relative change: b moved {rel_b:.1e}, x moved {rel_x:.1e}")
print(f"amplification ~ {rel_x / rel_b:.1e} (about kappa)")
assert rel_x / rel_b > 1e6
assert np.allclose(x0, [1.0, 1.0], atol=1e-6)
digits_left = 16 - math.log10(kappa)
print(f"digits of x you can trust ~ 16 - {math.log10(kappa):.0f} = {digits_left:.0f}")

print("\nAll linear-systems checks passed.")
