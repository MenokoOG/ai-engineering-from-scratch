"""Lesson 03: Matrix Transformations — rotate/scale/shear, power iteration, eigendecomposition, RNN explode/vanish."""

import math
import numpy as np


# ---------- from-scratch helpers ----------

def matmul(A, B):
    m, n, p = len(A), len(B), len(B[0])
    assert len(A[0]) == n
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
            for i in range(m)]


def matvec(A, v):
    return [sum(a * x for a, x in zip(row, v)) for row in A]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def scaling(sx, sy):
    return [[sx, 0.0], [0.0, sy]]


def shear(k):
    return [[1.0, k], [0.0, 1.0]]


def power_iteration(A, steps=200):
    """Repeatedly apply A and re-normalize; converges to the dominant eigenvector."""
    v = [1.0] * len(A)
    for _ in range(steps):
        v = matvec(A, v)
        n = norm(v)
        v = [x / n for x in v]
    eigenvalue = sum(x * y for x, y in zip(v, matvec(A, v)))  # Rayleigh quotient
    return eigenvalue, v


# ---------- demos ----------

def demo_transforms_order():
    print("=== 1. 2D transforms: rotation, scaling, shear — and order matters ===")
    R = rotation(math.pi / 2)          # rotate 90 degrees counterclockwise
    S = scaling(2.0, 1.0)              # stretch x by 2
    H = shear(1.0)                     # slant tops to the right

    p = [1.0, 0.0]
    print(f"  point {p}:")
    print(f"    rotate 90deg -> {[round(x, 3) for x in matvec(R, p)]}")
    print(f"    scale (2,1)  -> {matvec(S, p)}")
    print(f"    shear k=1 on [1,1] -> {matvec(H, [1.0, 1.0])}")

    RS = matmul(R, S)   # scale first, then rotate
    SR = matmul(S, R)   # rotate first, then scale
    assert np.allclose(RS, np.array(R) @ np.array(S))
    v = [1.0, 0.0]
    a, b = matvec(RS, v), matvec(SR, v)
    print(f"  R@S applied to {v} = {[round(x,3) for x in a]}  (scale then rotate)")
    print(f"  S@R applied to {v} = {[round(x,3) for x in b]}  (rotate then scale)")
    assert not np.allclose(a, b)
    print("  -> R@S != S@R: matrices apply right-to-left, and order changes the result")
    print()


def demo_power_iteration():
    print("=== 2. Power iteration from scratch (dominant eigenvector) ===")
    A = [[2.0, 1.0], [1.0, 2.0]]
    lam, v = power_iteration(A)
    np_vals, np_vecs = np.linalg.eigh(np.array(A))
    np_lam = np_vals[-1]
    np_v = np_vecs[:, -1]
    assert abs(lam - np_lam) < 1e-8
    assert np.allclose(np.abs(v), np.abs(np_v), atol=1e-6)  # sign is arbitrary
    print(f"  A = {A}")
    print(f"  dominant eigenvalue = {lam:.6f}   (numpy: {np_lam:.6f})")
    print(f"  dominant eigenvector = {[round(x, 6) for x in v]}   (numpy agrees up to sign)")
    Av = matvec(A, v)
    ratio = [a / x for a, x in zip(Av, v)]
    print(f"  check: A@v / v = {[round(r, 6) for r in ratio]} -> A only stretches v, never turns it")
    print()


def demo_eigendecomposition():
    print("=== 3. Eigendecomposition of [[2,1],[1,2]] with verification ===")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    vals, V = np.linalg.eig(A)
    D = np.diag(vals)
    reconstructed = V @ D @ np.linalg.inv(V)
    assert np.allclose(reconstructed, A)
    assert np.allclose(sorted(vals), [1.0, 3.0])
    print(f"  eigenvalues = {sorted(vals.tolist())}")
    print(f"  V @ D @ V^-1 =\n{reconstructed}")
    print("  -> matches A exactly: A is 'rotate into eigen-axes, stretch by 3 and 1, rotate back'")
    for i in range(2):
        assert np.allclose(A @ V[:, i], vals[i] * V[:, i])
    print("  check: A @ v_i == lambda_i * v_i for both eigenvectors")
    print()


def demo_rnn_explode_vanish():
    print("=== 4. Why RNN weights explode (eigenvalue > 1) or vanish (< 1) ===")
    h0 = [1.0, 1.0]
    for lam, label in [(1.2, "explodes"), (0.8, "vanishes")]:
        W = scaling(lam, lam)  # simple W with both eigenvalues = lam
        h = h0[:]
        sizes = []
        for step in range(1, 31):
            h = matvec(W, h)
            if step in (1, 10, 20, 30):
                sizes.append((step, norm(h)))
        printable = ", ".join(f"step {s}: {n:.4g}" for s, n in sizes)
        print(f"  eigenvalues = {lam}: {printable}  -> signal {label}")
        expected = norm(h0) * lam ** 30
        assert abs(sizes[-1][1] - expected) < 1e-6 * max(1.0, expected)

    # non-diagonal example: growth still follows the dominant eigenvalue
    A = [[2.0, 1.0], [1.0, 2.0]]  # eigenvalues 3 and 1
    h = [1.0, 0.0]
    for _ in range(20):
        h = matvec(A, h)
    growth = norm(h) / (3.0 ** 20)
    assert 0.1 < growth < 10.0
    print(f"  [[2,1],[1,2]] after 20 steps: |h| ~ 3^20 * {growth:.3f} -> dominant eigenvalue 3 takes over")
    print("  -> repeated multiplication raises eigenvalues to a power; only |lambda| = 1 is stable")
    print()


def main():
    demo_transforms_order()
    demo_power_iteration()
    demo_eigendecomposition()
    demo_rnn_explode_vanish()
    print("All checks passed.")


if __name__ == "__main__":
    main()
