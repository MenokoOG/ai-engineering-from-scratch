"""Lesson 06: Probability and Distributions - from scratch demos."""
import math
import numpy as np

rng = np.random.default_rng(42)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def text_hist(values, bins=20, width=40):
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max()
    for c, lo in zip(counts, edges[:-1]):
        bar = "#" * int(width * c / peak)
        print(f"  {lo:7.3f} | {bar}")


# ---------------------------------------------------------------
section("1) PMF vs PDF: dice vs gaussian")

# PMF: fair die. Discrete. Each outcome has a real probability.
die_pmf = {face: 1 / 6 for face in range(1, 7)}
print("Fair die PMF: P(X=k) = 1/6 for k = 1..6")
for face, p in die_pmf.items():
    print(f"  P(X={face}) = {p:.4f}")
total = sum(die_pmf.values())
print(f"  PMF sums to {total:.6f}  (must be exactly 1)")
assert abs(total - 1.0) < 1e-12


def gaussian_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# PDF: density, not probability. Can exceed 1. P(X = exact value) = 0.
mu, sigma = 0.0, 0.1
print(f"\nGaussian PDF with mu={mu}, sigma={sigma} (a narrow bell):")
print(f"  density at x=0: {gaussian_pdf(0, mu, sigma):.4f}  <- bigger than 1! Density is not probability.")
xs = np.linspace(-1, 1, 200001)
area = np.trapezoid([gaussian_pdf(x, mu, sigma) for x in xs], xs)
print(f"  area under the curve: {area:.6f}  (this is what must equal 1)")
assert abs(area - 1.0) < 1e-6
lo, hi = -0.1, 0.1
xs2 = np.linspace(lo, hi, 20001)
p_range = np.trapezoid([gaussian_pdf(x, mu, sigma) for x in xs2], xs2)
print(f"  P({lo} < X < {hi}) = area on that slice = {p_range:.4f}")
print("  P(X = 0.05 exactly) = 0. Only ranges have probability.")

# ---------------------------------------------------------------
section("2) Central Limit Theorem: means of uniform samples")

flat = rng.uniform(0, 1, 20000)
print("Raw uniform draws (flat shape):")
text_hist(flat, bins=12)

n = 30
means = rng.uniform(0, 1, (20000, n)).mean(axis=1)
print(f"\nMeans of {n} uniform draws each (bell shape appears):")
text_hist(means, bins=12)

pred_std = math.sqrt(1 / 12) / math.sqrt(n)  # uniform variance is 1/12
print(f"\nCLT predicts std of the means = sigma/sqrt(n) = {pred_std:.5f}")
print(f"Observed std of the means      = {means.std():.5f}")
assert abs(means.std() - pred_std) < 0.003

# ---------------------------------------------------------------
section("3) Stable softmax: the max-subtraction trick")


def naive_softmax(logits):
    e = np.exp(logits)
    return e / e.sum()


def stable_softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


big_logits = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = naive_softmax(big_logits)
print(f"logits = {big_logits}")
print(f"naive softmax:  {naive}   <- exp(1000) overflowed to inf, giving nan")
assert np.isnan(naive).any()

stable = stable_softmax(big_logits)
print(f"stable softmax: {np.round(stable, 4)}   <- subtract max first, same math, no overflow")
assert not np.isnan(stable).any()
assert abs(stable.sum() - 1.0) < 1e-9

small_logits = np.array([1.0, 2.0, 3.0])
assert np.allclose(naive_softmax(small_logits), stable_softmax(small_logits))
print("On safe logits, naive and stable give identical answers (checked).")

from scipy.special import softmax as scipy_softmax
assert np.allclose(stable, scipy_softmax(big_logits))
print("Matches scipy.special.softmax (checked).")

# ---------------------------------------------------------------
section("4) Cross-entropy loss = -log(p_true)")


def cross_entropy(probs, true_idx):
    return -math.log(probs[true_idx])


print("Loss only cares about the probability given to the CORRECT class.")
print(f"{'p(true class)':>14} | {'loss = -log(p)':>14}")
for p in [0.99, 0.9, 0.5, 0.1, 0.01, 0.001]:
    probs = [p, 1 - p]
    print(f"{p:>14} | {cross_entropy(probs, 0):>14.4f}")
print("Confident and right -> tiny loss. Confident and wrong -> huge loss.")
assert cross_entropy([1.0, 0.0], 0) == 0.0
assert abs(cross_entropy([0.5, 0.5], 0) - math.log(2)) < 1e-12

# ---------------------------------------------------------------
section("5) Why log-probs: products underflow, sums of logs do not")

p = 0.1
n_tokens = 1000
prod = 1.0
for _ in range(n_tokens):
    prod *= p
print(f"Multiplying {n_tokens} probabilities of {p}:")
print(f"  raw product      = {prod}   <- underflowed to exactly 0.0, all info lost")
assert prod == 0.0

log_sum = sum(math.log(p) for _ in range(n_tokens))
print(f"  sum of log-probs = {log_sum:.2f}   <- perfectly fine number")
assert abs(log_sum - n_tokens * math.log(p)) < 1e-6

log_a = 900 * math.log(0.1)
log_b = 1000 * math.log(0.1)
print(f"\nComparing two sequences: log-probs {log_a:.0f} vs {log_b:.0f}")
print("As raw products both would be 0.0 and impossible to compare.")
print("In log space the first is clearly more likely (checked).")
assert log_a > log_b

print("\nAll checks passed.")
