"""Lesson 02: Vectors & Matrices Operations — matmul, broadcasting, determinant, one NN layer."""

import numpy as np


# ---------- from-scratch ops ----------

def matmul(A, B):
    """Triple-loop matrix multiply: (m x n) @ (n x p) -> (m x p)."""
    m, n, p = len(A), len(B), len(B[0])
    assert len(A[0]) == n, f"inner dims must match: {len(A[0])} vs {n}"
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def elementwise_mul(A, B):
    assert len(A) == len(B) and len(A[0]) == len(B[0]), "shapes must match exactly"
    return [[a * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def broadcast_shapes(shape_a, shape_b):
    """Numpy's broadcasting rule: align from the right; dims must match or be 1.
    Returns the result shape, or None if the shapes are incompatible."""
    result = []
    for da, db in zip(reversed((1,) * max(0, len(shape_b) - len(shape_a)) + tuple(shape_a)),
                      reversed((1,) * max(0, len(shape_a) - len(shape_b)) + tuple(shape_b))):
        if da == db or da == 1 or db == 1:
            result.append(max(da, db))
        else:
            return None
    return tuple(reversed(result))


def det2(M):
    (a, b), (c, d) = M
    return a * d - b * c


def det3(M):
    """Expansion along the first row (cofactors)."""
    (a, b, c), (d, e, f), (g, h, i) = M
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def relu(v):
    return [x if x > 0 else 0.0 for x in v]


def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]


# ---------- demos ----------

def demo_matmul_and_identity():
    print("=== 1. Matmul from scratch (triple loop) + identity matrix ===")
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]          # 2x3
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]     # 3x2
    C = matmul(A, B)                                # 2x2
    assert np.allclose(C, np.array(A) @ np.array(B))
    print(f"  (2x3) @ (3x2) = {C}   (matches numpy)")

    I = identity(3)
    assert matmul(A, I) == A
    assert np.allclose(identity(4), np.eye(4))
    print("  A @ I == A  -> identity is the 'multiply by 1' of matrices")

    try:
        matmul(B, B)
    except AssertionError as e:
        print(f"  (3x2) @ (3x2) correctly rejected: {e}")
    print()


def demo_elementwise_vs_matmul():
    print("=== 2. Element-wise product vs matrix product ===")
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    ew = elementwise_mul(A, B)
    mm = matmul(A, B)
    assert np.allclose(ew, np.array(A) * np.array(B))
    assert np.allclose(mm, np.array(A) @ np.array(B))
    print(f"  A * B (element-wise) = {ew}")
    print(f"  A @ B (matrix)       = {mm}")
    print("  -> element-wise pairs up cells; matmul mixes rows with columns")
    print()


def demo_broadcasting():
    print("=== 3. Broadcasting rule checker (from scratch) ===")
    cases = [
        ((3, 4), (4,)),
        ((3, 4), (3, 1)),
        ((8, 1, 6, 1), (7, 1, 5)),
        ((3, 4), (2, 4)),
        ((5,), (5,)),
    ]
    for sa, sb in cases:
        ours = broadcast_shapes(sa, sb)
        try:
            np_shape = tuple(np.broadcast_shapes(sa, sb))
        except ValueError:
            np_shape = None
        assert ours == np_shape, (sa, sb, ours, np_shape)
        verdict = f"-> {ours}" if ours else "-> INCOMPATIBLE"
        print(f"  {sa} with {sb} {verdict}   (matches numpy)")
    print("  rule: align shapes from the right; each dim must match or be 1")
    print()


def demo_determinant():
    print("=== 4. Determinant (2x2, 3x3) and what det=0 means ===")
    M2 = [[3.0, 1.0], [2.0, 4.0]]
    M3 = [[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [1.0, 1.0, 4.0]]
    assert abs(det2(M2) - np.linalg.det(np.array(M2))) < 1e-9
    assert abs(det3(M3) - np.linalg.det(np.array(M3))) < 1e-9
    print(f"  det{M2} = {det2(M2)}   (matches numpy)")
    print(f"  det(3x3 example) = {det3(M3)}   (matches numpy)")

    S = [[1.0, 2.0], [2.0, 4.0]]  # second row = 2 * first row
    d = det2(S)
    assert d == 0.0
    print(f"  det{S} = {d} -> rows are dependent: the matrix squashes 2D onto a line")
    print("  det=0 means information is destroyed, so the matrix cannot be inverted")
    print()


def demo_nn_layer():
    print("=== 5. One neural-net layer: relu(W @ x + b) with my own ops ===")
    W = [[0.5, -1.0, 2.0],
         [1.5, 0.5, -0.5],
         [-2.0, 1.0, 1.0],
         [0.1, 0.2, 0.3]]              # 4x3: 3 inputs -> 4 outputs
    x = [1.0, 2.0, -1.0]
    b = [0.1, -3.0, 0.2, 0.5]

    Wx = [row[0] for row in matmul(W, [[v] for v in x])]  # 4x3 @ 3x1 -> 4x1
    out = relu(add_vec(Wx, b))

    np_out = np.maximum(np.array(W) @ np.array(x) + np.array(b), 0.0)
    assert np.allclose(out, np_out)
    print(f"  x = {x}")
    print(f"  W @ x + b = {[round(v, 3) for v in add_vec(Wx, b)]}")
    print(f"  relu(...) = {out}   (matches numpy)")
    print("  -> W mixes the inputs, b shifts them, relu zeroes out negatives")

    # broadcasting in the real thing: a batch X (2x3) plus bias b (4,)
    X = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]])
    batch_out = np.maximum(X @ np.array(W).T + np.array(b), 0.0)
    assert broadcast_shapes((2, 4), (4,)) == (2, 4)
    assert batch_out.shape == (2, 4)
    print(f"  batch version: (2,4) + bias (4,) broadcasts to (2,4) -> same bias added to every row")
    print()


def main():
    demo_matmul_and_identity()
    demo_elementwise_vs_matmul()
    demo_broadcasting()
    demo_determinant()
    demo_nn_layer()
    print("All checks passed.")


if __name__ == "__main__":
    main()
