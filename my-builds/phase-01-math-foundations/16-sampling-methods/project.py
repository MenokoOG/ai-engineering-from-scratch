"""Lesson 16: Sampling Methods - from scratch demos."""
import math
import numpy as np

rng = np.random.default_rng(16)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "pizza"]
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])

# ---------------------------------------------------------------
section("1) Temperature scaling of a toy LM distribution")

print(f"Toy vocab: {vocab}")
for T in [0.5, 1.0, 2.0, 100.0]:
    p = softmax(logits / T)
    top = p.max()
    print(f"  T={T:>5}: top prob={top:.3f}, entropy={entropy(p):.3f}, "
          f"probs={np.round(p, 3)}")
p_cold = softmax(logits / 0.5)
p_base = softmax(logits)
p_hot = softmax(logits / 100.0)
print("T<1 sharpens (rich get richer), T>1 flattens toward uniform.")
assert p_cold.max() > p_base.max() > p_hot.max()
assert entropy(p_cold) < entropy(p_base) < entropy(p_hot)
assert np.allclose(p_hot, 1 / len(vocab), atol=0.02)  # T=100 is nearly uniform
assert np.argmax(p_cold) == np.argmax(p_hot)  # ranking never changes

# ---------------------------------------------------------------
section("2) Top-k vs top-p (nucleus) sampling")

p = softmax(logits)
order = np.argsort(p)[::-1]


def top_k_filter(p, k):
    keep = np.argsort(p)[::-1][:k]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum(), len(keep)


def top_p_filter(p, p_threshold):
    idx = np.argsort(p)[::-1]
    csum = np.cumsum(p[idx])
    cut = int(np.searchsorted(csum, p_threshold)) + 1
    keep = idx[:cut]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum(), cut


qk, nk = top_k_filter(p, 3)
qp, np_kept = top_p_filter(p, 0.8)
print(f"Full distribution: {[f'{w}:{pi:.3f}' for w, pi in zip(vocab, p)]}")
print(f"top-k (k=3): always keeps {nk} tokens -> {[vocab[i] for i in np.flatnonzero(qk)]}")
print(f"top-p (p=0.8): keeps smallest set covering 80% -> {np_kept} tokens")
assert abs(qk.sum() - 1) < 1e-9 and abs(qp.sum() - 1) < 1e-9

flat_p = softmax(logits / 100.0)   # confused model: nearly uniform
peaky_p = softmax(logits * 3)      # confident model: very peaked
_, n_flat = top_p_filter(flat_p, 0.8)
_, n_peaky = top_p_filter(peaky_p, 0.8)
print(f"\ntop-p adapts, top-k does not:")
print(f"  confident distribution: top-p keeps {n_peaky} token(s)")
print(f"  flat distribution:      top-p keeps {n_flat} token(s)")
print(f"  top-k keeps exactly k tokens either way.")
assert n_peaky < n_flat

# ---------------------------------------------------------------
section("3) Reparameterization trick: z = mu + sigma*eps")

mu, sigma = 1.5, 2.0
n = 200000
# Goal: gradient of E[z^2] w.r.t. mu, where z ~ N(mu, sigma^2). True answer: 2*mu.
print("Goal: d/dmu of E[z^2] where z ~ Normal(mu, sigma). True answer = 2*mu =", 2 * mu)
print("Problem: 'sample from N(mu, sigma)' is a dice roll. No gradient flows through dice.")
eps = rng.standard_normal(n)
z = mu + sigma * eps           # randomness moved into eps; mu is now plain arithmetic
grad_est = (2 * z).mean()      # dz/dmu = 1, so d(z^2)/dmu = 2z per sample
print(f"Reparameterized: z = mu + sigma*eps, eps ~ N(0,1). Now dz/dmu = 1 exactly.")
print(f"  Monte Carlo gradient estimate: {grad_est:.4f} (true: {2*mu})")
assert abs(grad_est - 2 * mu) < 0.05

h = 1e-3
same_eps = mu + h + sigma * eps
fd = ((same_eps ** 2).mean() - (z ** 2).mean()) / h
print(f"  Finite-difference check (same eps, nudge mu): {fd:.4f}")
assert abs(fd - 2 * mu) < 0.05
print("Same noise, nudged mu -> smooth change. Gradients can flow. This powers VAEs.")

# ---------------------------------------------------------------
section("4) Metropolis-Hastings MCMC on a 1D bimodal distribution")


def target(x):  # two bumps: N(-3,1) and N(3,1), equal weight (unnormalized ok)
    return math.exp(-0.5 * (x + 3) ** 2) + math.exp(-0.5 * (x - 3) ** 2)


def metropolis(proposal_std, steps=20000):
    x = 0.0
    samples = np.empty(steps)
    accepts = 0
    for i in range(steps):
        prop = x + rng.normal(0, proposal_std)
        if rng.uniform() < target(prop) / target(x):
            x = prop
            accepts += 1
        samples[i] = x
    return samples, accepts / steps


print("Target: two bumps at -3 and +3. Sampler only needs target(x) up to a constant.")
for std in [0.1, 1.0, 2.5, 50.0]:
    s, rate = metropolis(std)
    left = (s < 0).mean()
    print(f"  proposal_std={std:>5}: acceptance rate={rate:.3f}, "
          f"sample mean={s.mean():+.2f}, fraction in left bump={left:.2f}")
s_good, rate_good = metropolis(2.5)
s_huge, rate_huge = metropolis(50.0)
print("Tiny steps: high acceptance but crawls, can get stuck in one bump.")
print("Huge steps: proposals land in no-man's-land, almost all REJECTED ->")
print("the chain freezes in place, repeating the same value for long stretches.")
assert rate_huge < 0.1          # acceptance collapses with oversized proposals
assert rate_good > 0.3
assert 0.3 < (s_good < 0).mean() < 0.7   # good chain visits both bumps
assert abs(s_good.mean()) < 0.5

# ---------------------------------------------------------------
section("5) Rejection sampling and the curse of dimensionality")

print("Task: sample uniformly inside a d-dimensional ball by throwing darts")
print("into the enclosing cube and keeping darts that land inside the ball.")
n_darts = 100000
print(f"{'dim':>4} | {'acceptance rate':>15} | {'theory':>10}")
rates = {}
for d in [1, 2, 5, 10, 15]:
    darts = rng.uniform(-1, 1, (n_darts, d))
    inside = (darts ** 2).sum(axis=1) <= 1.0
    rate = inside.mean()
    theory = math.pi ** (d / 2) / math.gamma(d / 2 + 1) / 2 ** d
    rates[d] = rate
    print(f"{d:>4} | {rate:>15.5f} | {theory:>10.5f}")
print("Acceptance collapses exponentially with dimension: the ball occupies")
print("almost none of the cube. Rejection sampling is hopeless in high dims,")
print("which is exactly why MCMC and friends exist.")
assert abs(rates[2] - math.pi / 4) < 0.01
assert rates[1] > rates[2] > rates[5] > rates[10] >= rates[15]
assert rates[15] < 0.001

print("\nAll checks passed.")
