"""Lesson 09 - Information Theory, built from scratch with numpy."""
import numpy as np

rng = np.random.default_rng(9)
EPS = 1e-12


def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p * np.log(p + EPS)))


def cross_entropy(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(-np.sum(p * np.log(q + EPS)))


def kl_divergence(p, q):
    return cross_entropy(p, q) - entropy(p)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) ENTROPY: fair dice vs loaded dice")
fair = np.full(6, 1 / 6)
loaded = np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02])
certain = np.array([1.0, 0, 0, 0, 0, 0])

h_fair = entropy(fair)
h_loaded = entropy(loaded)
h_certain = entropy(certain)
print(f"fair die     H = {h_fair:.4f} nats (max possible = ln 6 = {np.log(6):.4f})")
print(f"loaded die   H = {h_loaded:.4f} nats (mostly predictable -> low)")
print(f"certain die  H = {h_certain:.4f} nats (no surprise at all)")
assert np.isclose(h_fair, np.log(6))
assert h_loaded < h_fair
assert np.isclose(h_certain, 0.0, atol=1e-9)
print("PASS: uniform gives max entropy, certainty gives zero entropy")

# ---------------------------------------------------------------
section("2) CROSS-ENTROPY and KL, identity CE = H(p) + KL(p||q)")
p = fair
q = loaded
ce = cross_entropy(p, q)
kl = kl_divergence(p, q)
print(f"true p = fair die, model q = loaded die")
print(f"H(p)          = {entropy(p):.6f}")
print(f"KL(p||q)      = {kl:.6f}")
print(f"CE(p,q)       = {ce:.6f}")
print(f"H(p)+KL(p||q) = {entropy(p) + kl:.6f}")
assert np.isclose(ce, entropy(p) + kl)
assert kl >= 0
assert np.isclose(kl_divergence(p, p), 0.0, atol=1e-9)
print("PASS: CE = H(p) + KL verified; KL >= 0; KL(p||p) = 0")

# ---------------------------------------------------------------
section("3) MINIMIZING CE = MINIMIZING KL (when p is fixed)")
print("KL(p||q) = CE(p,q) - H(p).  H(p) does not depend on q.")
print("So over candidate models q, CE and KL differ by a constant:")
print(f"{'model q':<28}{'CE(p,q)':>10}{'KL(p||q)':>10}{'CE-KL':>10}")
candidates = {
    "loaded [.9,...]": loaded,
    "mild   [.3,.14x5]": np.array([0.30, 0.14, 0.14, 0.14, 0.14, 0.14]),
    "exact fair": fair,
}
for name, qc in candidates.items():
    c, k = cross_entropy(p, qc), kl_divergence(p, qc)
    print(f"{name:<28}{c:>10.4f}{k:>10.4f}{c - k:>10.4f}")
    assert np.isclose(c - k, entropy(p))
print("PASS: CE - KL is always H(p); same q minimizes both")

# ---------------------------------------------------------------
section("4) PERPLEXITY = exp(cross-entropy) on a toy language model")
vocab = 50
true_words = rng.integers(0, vocab, size=2000)

# model A: uniform over 50 words -> perplexity should be exactly 50
logp_uniform = np.log(np.full(vocab, 1 / vocab) + EPS)
# model B: puts 60% mass on the actually-frequent half of the vocab
probs_b = np.where(np.arange(vocab) < 25, 0.60 / 25, 0.40 / 25)
logp_b = np.log(probs_b + EPS)


def perplexity(logp_table, words):
    avg_nll = -np.mean(logp_table[words])
    return float(np.exp(avg_nll)), float(avg_nll)


ppl_a, nll_a = perplexity(logp_uniform, true_words)
ppl_b, nll_b = perplexity(logp_b, true_words)
print(f"vocab size = {vocab}, test tokens = {len(true_words)}")
print(f"uniform model: avg CE = {nll_a:.4f}, perplexity = {ppl_a:.2f}")
print(f"skewed model : avg CE = {nll_b:.4f}, perplexity = {ppl_b:.2f}")
print("perplexity 50 = 'as confused as picking among 50 equally likely words'")
assert np.isclose(ppl_a, 50.0)
print("PASS: uniform model over 50 words has perplexity exactly 50")

# ---------------------------------------------------------------
section("5) MUTUAL INFORMATION vs PEARSON CORRELATION (y = x^2)")
n = 200_000
x = rng.uniform(-1, 1, size=n)
y = x ** 2

corr = float(np.corrcoef(x, y)[0, 1])


def mutual_information(a, b, bins=20):
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = joint / joint.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


mi_xy = mutual_information(x, y)
z = rng.uniform(-1, 1, size=n)  # independent control
mi_xz = mutual_information(x, z)
print(f"y = x^2 relationship (perfectly dependent, but not linear):")
print(f"  Pearson correlation(x, y) = {corr:+.4f}  (near zero: misses it)")
print(f"  Mutual information(x, y)  = {mi_xy:.4f} nats (clearly > 0: sees it)")
print(f"independent control z:")
print(f"  Mutual information(x, z)  = {mi_xz:.4f} nats (near zero, as it should be)")
assert abs(corr) < 0.02
assert mi_xy > 0.5
assert mi_xz < 0.05
print("PASS: correlation only detects linear links, MI detects any dependence")

print("\nALL CHECKS PASSED")
