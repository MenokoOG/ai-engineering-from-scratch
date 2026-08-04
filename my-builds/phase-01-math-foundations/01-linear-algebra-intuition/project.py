"""Lesson 01: Linear Algebra Intuition — a mini vector toolkit from scratch."""

import math
import numpy as np


# ---------- from-scratch vector toolkit (pure Python lists) ----------

def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def scale(a, s):
    return [x * s for x in a]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


# ---------- from-scratch matrix helpers ----------

def matmul(A, B):
    rows, inner, cols = len(A), len(B), len(B[0])
    assert len(A[0]) == inner
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(rows)]


def rank(M, tol=1e-10):
    """Row-reduce a copy of M and count nonzero rows (Gaussian elimination)."""
    M = [row[:] for row in M]
    n_rows, n_cols = len(M), len(M[0])
    r = 0
    for col in range(n_cols):
        pivot = None
        for i in range(r, n_rows):
            if abs(M[i][col]) > tol:
                pivot = i
                break
        if pivot is None:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        M[r] = scale(M[r], 1.0 / M[r][col])
        for i in range(n_rows):
            if i != r and abs(M[i][col]) > tol:
                M[i] = add(M[i], scale(M[r], -M[i][col]))
        r += 1
        if r == n_rows:
            break
    return r


def linearly_independent(vectors):
    """Vectors are independent iff the rank equals the number of vectors."""
    return rank([list(v) for v in vectors]) == len(vectors)


# ---------- demos ----------

def demo_dot_and_cosine():
    print("=== 1. Dot product and cosine similarity (from scratch) ===")
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    d = dot(a, b)
    c = cosine_similarity(a, b)
    assert abs(d - np.dot(a, b)) < 1e-12
    np_cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(c - np_cos) < 1e-12
    print(f"dot({a}, {b}) = {d}   (matches numpy)")
    print(f"cosine similarity = {c:.4f}   (matches numpy)")
    print()


def demo_word_embeddings():
    print("=== 2. Cosine similarity as 'meaning similarity' (toy embeddings) ===")
    # made-up 4-dim embeddings: [animal-ness, size, royalty, food-ness]
    emb = {
        "cat":    [0.9, 0.2, 0.0, 0.1],
        "dog":    [0.9, 0.3, 0.0, 0.1],
        "king":   [0.1, 0.5, 0.9, 0.0],
        "queen":  [0.1, 0.4, 0.9, 0.0],
        "pizza":  [0.0, 0.2, 0.0, 0.9],
    }
    pairs = [("cat", "dog"), ("king", "queen"), ("cat", "pizza"), ("king", "pizza")]
    for w1, w2 in pairs:
        print(f"  sim({w1:5s}, {w2:5s}) = {cosine_similarity(emb[w1], emb[w2]):.3f}")
    assert cosine_similarity(emb["cat"], emb["dog"]) > cosine_similarity(emb["cat"], emb["pizza"])
    assert cosine_similarity(emb["king"], emb["queen"]) > cosine_similarity(emb["king"], emb["pizza"])
    print("  -> similar words score high, unrelated words score low")
    print()


def demo_independence_and_rank():
    print("=== 3. Linear independence and matrix rank (from scratch) ===")
    v1, v2, v3 = [1, 0, 0], [0, 1, 0], [2, 1, 0]
    vecs = [v1, v2, v3]
    indep = linearly_independent(vecs)
    r = rank([list(v) for v in vecs])
    np_r = np.linalg.matrix_rank(np.array(vecs, dtype=float))
    assert r == np_r == 2
    print(f"  vectors: {v1}, {v2}, {v3}")
    print(f"  rank = {r} (numpy agrees: {np_r}) -> NOT independent (v3 = 2*v1 + v2)")

    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert linearly_independent(basis)
    print(f"  standard basis rank = {rank(basis)} -> independent")

    M = [[1.0, 2.0], [2.0, 4.0]]
    r2 = rank(M)
    assert r2 == np.linalg.matrix_rank(np.array(M)) == 1
    print(f"  [[1,2],[2,4]] rank = {r2} -> rows are copies, matrix carries 1D of info")
    print()


def demo_lora():
    print("=== 4. LoRA-style low-rank decomposition: W ~= A @ B ===")
    rng = np.random.default_rng(0)
    d, r = 8, 2
    # build a weight matrix that is secretly rank-2
    A_true = rng.normal(size=(d, r))
    B_true = rng.normal(size=(r, d))
    W = A_true @ B_true

    # recover a rank-2 factorization with SVD (the "ideal LoRA")
    U, S, Vt = np.linalg.svd(W)
    A = U[:, :r] * S[:r]
    B = Vt[:r, :]
    W_approx = A @ B

    err = np.max(np.abs(W - W_approx))
    assert err < 1e-10
    full_params = d * d
    lora_params = d * r + r * d
    print(f"  W is {d}x{d} = {full_params} numbers")
    print(f"  A ({d}x{r}) and B ({r}x{d}) = {lora_params} numbers, {lora_params/full_params:.0%} of full size")
    print(f"  max reconstruction error: {err:.2e} -> W == A@B (W was truly rank {r})")

    # verify our from-scratch matmul agrees with numpy on A @ B
    ours = matmul(A.tolist(), B.tolist())
    assert np.allclose(ours, W_approx)
    print("  from-scratch matmul(A, B) matches numpy A @ B")
    print()


def main():
    demo_dot_and_cosine()
    demo_word_embeddings()
    demo_independence_and_rank()
    demo_lora()
    print("All checks passed.")


if __name__ == "__main__":
    main()
