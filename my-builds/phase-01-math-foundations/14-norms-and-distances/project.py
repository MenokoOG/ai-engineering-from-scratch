"""Lesson 14: Norms and distances. Different rulers for 'how big' and 'how far'."""
import math
import numpy as np

np.random.seed(14)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) L1, L2, Linf norms from scratch")


def l1_norm(v):
    return sum(abs(x) for x in v)


def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))


def linf_norm(v):
    return max(abs(x) for x in v)


v = [3.0, -4.0, 1.0]
print(f"v = {v}")
print(f"L1  (sum of abs)      = {l1_norm(v)}")
print(f"L2  (straight length) = {l2_norm(v):.6f}")
print(f"Linf (biggest entry)  = {linf_norm(v)}")
vn = np.array(v)
assert math.isclose(l1_norm(v), np.linalg.norm(vn, 1))
assert math.isclose(l2_norm(v), np.linalg.norm(vn, 2))
assert math.isclose(linf_norm(v), np.linalg.norm(vn, np.inf))
print("all three match numpy.linalg.norm")

# ---------------------------------------------------------------
section("2) Manhattan vs Euclidean intuition")

start, goal = (0, 0), (3, 4)
diff = [goal[0] - start[0], goal[1] - start[1]]
print(f"walk from {start} to {goal} on a city grid")
print(f"Manhattan (L1) distance = {l1_norm(diff)}  (blocks walked: 3 east + 4 north)")
print(f"Euclidean (L2) distance = {l2_norm(diff)}  (crow-flies diagonal)")
assert l1_norm(diff) == 7 and l2_norm(diff) == 5
print("L1 >= L2 always: streets are never shorter than flying")

# ---------------------------------------------------------------
section("3) cosine similarity on toy embeddings")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine_sim(a, b):
    return dot(a, b) / (l2_norm(a) * l2_norm(b))


def euclidean(a, b):
    return l2_norm([x - y for x, y in zip(a, b)])


emb = {
    "cat":       [2.0, 1.8, 0.1],
    "kitten":    [1.9, 2.0, 0.2],
    "CAT_ESSAY": [20.0, 18.0, 1.0],   # same topic, 10x "longer document"
    "car":       [0.2, 0.1, 2.0],
}
print("toy embeddings (axis meaning: [animal-ness, pet-ness, machine-ness])")
pairs = [("cat", "kitten"), ("cat", "CAT_ESSAY"), ("cat", "car")]
for a, b in pairs:
    print(f"  {a:>9} vs {b:<9}: euclid = {euclidean(emb[a], emb[b]):7.3f},"
          f"  cosine = {cosine_sim(emb[a], emb[b]):6.3f}")
assert euclidean(emb["cat"], emb["CAT_ESSAY"]) > euclidean(emb["cat"], emb["car"])
assert cosine_sim(emb["cat"], emb["CAT_ESSAY"]) > cosine_sim(emb["cat"], emb["car"])
print("raw distance says the long cat essay is 'farther' than a car (wrong!)")
print("cosine ignores length and sees the shared direction = shared meaning")
assert math.isclose(
    cosine_sim(emb["cat"], emb["kitten"]),
    float(np.dot(emb["cat"], emb["kitten"])
          / (np.linalg.norm(emb["cat"]) * np.linalg.norm(emb["kitten"]))),
)

# ---------------------------------------------------------------
section("4) L1 makes sparse weights, L2 does not (tiny regression)")

n, d = 200, 10
X = np.random.randn(n, d)
w_true = np.zeros(d)
w_true[:3] = [3.0, -2.0, 1.5]          # only 3 features actually matter
y = X @ w_true + 0.1 * np.random.randn(n)


def train(reg, lam=0.4, lr=0.01, steps=3000):
    w = np.zeros(d)
    for _ in range(steps):
        grad = 2 * X.T @ (X @ w - y) / n
        if reg == "l1":
            grad = grad + lam * np.sign(w)      # subgradient of lam*|w|
        else:
            grad = grad + 2 * lam * w           # gradient of lam*w^2
        w -= lr * grad
    return w


w_l1 = train("l1")
w_l2 = train("l2")
tol = 1e-2
print(f"true weights          : {np.round(w_true, 3)}")
print(f"L1-trained weights    : {np.round(w_l1, 3)}")
print(f"L2-trained weights    : {np.round(w_l2, 3)}")
zeros_l1 = int(np.sum(np.abs(w_l1) < tol))
zeros_l2 = int(np.sum(np.abs(w_l2) < tol))
print(f"weights ~zero (|w|<{tol}) : L1 -> {zeros_l1}/10, L2 -> {zeros_l2}/10")
assert zeros_l1 >= 6
assert zeros_l2 <= 2
print("L1's constant pull snaps useless weights to zero; L2's pull fades near zero")

# ---------------------------------------------------------------
section("5) Mahalanobis distance on correlated data")

cov_true = np.array([[1.0, 0.95], [0.95, 1.0]])   # strongly correlated x,y
L = np.linalg.cholesky(cov_true)
data = (L @ np.random.randn(2, 500)).T
mean = data.mean(axis=0)
diffs = data - mean
cov = diffs.T @ diffs / (len(data) - 1)            # covariance from scratch


def inv_2x2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return np.array([[m[1][1], -m[0][1]], [-m[1][0], m[0][0]]]) / det


cov_inv = inv_2x2(cov)
assert np.allclose(cov_inv, np.linalg.inv(cov))


def mahalanobis(p, mu, cov_inv):
    d = p - mu
    return math.sqrt(d @ cov_inv @ d)


p_along = mean + np.array([2.0, 2.0])     # far in space, but along the data trend
p_across = mean + np.array([1.0, -1.0])   # nearby in space, but against the trend
print("data cloud: x and y move together (correlation 0.95)")
for name, p in [("along-trend (2, 2)", p_along), ("across-trend (1,-1)", p_across)]:
    e = np.linalg.norm(p - mean)
    m = mahalanobis(p, mean, cov_inv)
    print(f"  point {name}: euclidean = {e:.2f}, mahalanobis = {m:.2f}")
e_along, m_along = np.linalg.norm(p_along - mean), mahalanobis(p_along, mean, cov_inv)
e_across, m_across = np.linalg.norm(p_across - mean), mahalanobis(p_across, mean, cov_inv)
assert e_across < e_along          # Euclidean: across-trend point looks closer
assert m_across > m_along          # Mahalanobis: it is actually the weird one
print("the euclidean-closer point is the true outlier once correlation is considered")

# ---------------------------------------------------------------
section("6) Wasserstein vs KL for non-overlapping distributions")

bins = np.arange(10)


def make_dist(where):
    p = np.zeros(10)
    p[where] = 0.5
    return p


p = make_dist([0, 1])          # mass on the left
q_near = make_dist([2, 3])     # just to the right, no overlap
q_far = make_dist([8, 9])      # far right, no overlap


def kl(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi == 0:
                return float("inf")
            total += pi * math.log(pi / qi)
    return total


def wasserstein_1d(p, q):
    # earth mover's distance = area between the two running totals (CDFs)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


print(f"p mass at bins [0,1]; q_near at [2,3]; q_far at [8,9] (no overlap with p)")
print(f"KL(p, q_near) = {kl(p, q_near)},  KL(p, q_far) = {kl(p, q_far)}")
print(f"W(p, q_near)  = {wasserstein_1d(p, q_near)},  W(p, q_far)  = {wasserstein_1d(p, q_far)}")
assert math.isinf(kl(p, q_near)) and math.isinf(kl(p, q_far))
assert wasserstein_1d(p, q_far) > wasserstein_1d(p, q_near)
print("KL just says 'infinitely wrong' for both -> useless gradient signal")
print("Wasserstein measures how far the dirt must move -> near < far, usable signal")

print("\nAll norms-and-distances checks passed.")
