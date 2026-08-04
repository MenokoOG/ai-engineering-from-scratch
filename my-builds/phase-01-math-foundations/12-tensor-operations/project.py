"""Lesson 12 - Tensor Operations: shapes, broadcasting, einsum, layouts, contiguity."""
import numpy as np

rng = np.random.default_rng(12)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) SHAPE AND RANK WALKTHROUGH")
examples = [
    ("scalar (loss value)", np.array(3.5)),
    ("vector (one embedding)", np.zeros(768)),
    ("matrix (batch of embeddings)", np.zeros((32, 768))),
    ("3D (batch, seq_len, embed)", np.zeros((32, 128, 768))),
    ("4D (batch, heads, seq, head_dim)", np.zeros((32, 12, 128, 64))),
]
print(f"{'meaning':<34}{'shape':<22}{'rank':>5}{'elements':>12}")
for name, t in examples:
    print(f"{name:<34}{str(t.shape):<22}{t.ndim:>5}{t.size:>12}")
print("rank = number of axes = len(shape). elements = product of shape.")

# ---------------------------------------------------------------
section("2) BROADCASTING RULE, WRITTEN FROM SCRATCH")


def broadcast_shape(shape_a, shape_b):
    """Right-align both shapes; each pair must match or contain a 1."""
    a, b = list(shape_a), list(shape_b)
    while len(a) < len(b):
        a.insert(0, 1)
    while len(b) < len(a):
        b.insert(0, 1)
    out = []
    for da, db in zip(a, b):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"cannot broadcast {shape_a} with {shape_b}")
    return tuple(out)


cases = [
    ((8, 1, 6, 1), (7, 1, 5)),
    ((256, 256, 3), (3,)),
    ((5, 4), (1,)),
    ((15, 3, 5), (15, 1, 5)),
    ((2, 1), (1, 3)),
    ((32, 128, 768), (768,)),
]
for sa, sb in cases:
    ours = broadcast_shape(sa, sb)
    numpys = np.broadcast_shapes(sa, sb)
    assert ours == numpys
    print(f"{str(sa):>15} x {str(sb):<12} -> {ours}")
assert broadcast_shape((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)
try:
    broadcast_shape((3, 4), (2, 4))
    raise AssertionError("should have failed")
except ValueError as e:
    print(f"(3, 4) x (2, 4)      -> ValueError: {e}")
print("PASS: from-scratch rule matches np.broadcast_shapes on every case")

# ---------------------------------------------------------------
section("3) EINSUM 'bhtd,bhsd->bhts' (attention scores), loops vs numpy")
B, H, T, S, D = 2, 3, 4, 5, 6  # batch, heads, query len, key len, head dim
Q = rng.normal(size=(B, H, T, D))
K = rng.normal(size=(B, H, S, D))

scores_loops = np.zeros((B, H, T, S))
for b in range(B):
    for h in range(H):
        for t in range(T):
            for s_ in range(S):
                total = 0.0
                for d in range(D):  # d appears in both inputs, not output -> summed
                    total += Q[b, h, t, d] * K[b, h, s_, d]
                scores_loops[b, h, t, s_] = total

scores_einsum = np.einsum("bhtd,bhsd->bhts", Q, K)
scores_matmul = Q @ K.transpose(0, 1, 3, 2)
assert np.allclose(scores_loops, scores_einsum)
assert np.allclose(scores_loops, scores_matmul)
print(f"Q shape {Q.shape} (queries), K shape {K.shape} (keys)")
print(f"output shape {scores_einsum.shape} = one score per (query, key) pair")
print("index d is summed away (it is in the inputs but not the output);")
print("b, h, t, s survive. Same result as Q @ K.transpose(-1, -2).")
print("PASS: 5 nested loops == np.einsum == batched matmul")

# ---------------------------------------------------------------
section("4) NCHW IMAGE BATCH LAYOUT")
batch = rng.integers(0, 256, size=(4, 3, 8, 8)).astype(np.float32)  # N,C,H,W
print(f"batch shape {batch.shape} = (N=4 images, C=3 channels, H=8, W=8)")
print(f"image 2, red channel, pixel row 5 col 7 -> batch[2, 0, 5, 7] = {batch[2,0,5,7]:.0f}")
per_channel_mean = batch.mean(axis=(0, 2, 3))  # average over N, H, W
print(f"per-channel mean (like BatchNorm stats), shape {per_channel_mean.shape}:")
print(f"  {np.round(per_channel_mean, 2)}")
nhwc = batch.transpose(0, 2, 3, 1)
print(f"NHWC version (TensorFlow-style) has shape {nhwc.shape}")
assert nhwc.shape == (4, 8, 8, 3)
assert np.isclose(nhwc[2, 5, 7, 0], batch[2, 0, 5, 7])
print("PASS: PyTorch default is NCHW; same data, axes just reordered")

# ---------------------------------------------------------------
section("5) CONTIGUITY: why transpose-then-view fails in torch")
M = np.arange(6, dtype=np.int64).reshape(2, 3)
Mt = M.T
print(f"M (2x3), row-major memory: {M.ravel(order='K')}")
print(f"M strides  = {M.strides} bytes  (walk a row: jump 8; next row: jump 24)")
print(f"M.T strides= {Mt.strides} bytes (transpose = SAME memory, swapped strides)")
assert np.shares_memory(M, Mt)
print(f"M.T contiguous? {Mt.flags['C_CONTIGUOUS']}  <- reading M.T row by row")
print("   hops around memory: 0,3,1,4,2,5 instead of 0,1,2,3,4,5")

flat_wanted = Mt.reshape(-1)          # numpy: silently copies
assert not np.shares_memory(M, flat_wanted)
print(f"numpy Mt.reshape(-1) = {flat_wanted} -> made a COPY (new memory)")
print("torch .view() promises NO copy (just relabel the same memory),")
print("but no stride relabeling can turn 0,3,1,4,2,5 into a flat walk.")
print("So torch: Mt.view(-1) raises an error; fix = Mt.contiguous().view(-1)")
print("or Mt.reshape(-1), which copies when it must (like numpy).")

flat_contig = np.ascontiguousarray(Mt).reshape(-1)  # the .contiguous() fix
assert np.array_equal(flat_contig, flat_wanted)
assert np.array_equal(flat_wanted, np.array([0, 3, 1, 4, 2, 5]))
print("PASS: transpose shares memory; flattening it requires a copy")

print("\nALL CHECKS PASSED")
