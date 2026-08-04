"""Lesson 22: Stochastic processes from scratch.
Markov chain weather, random walk sqrt(n) law, stationary distribution two ways,
Langevin dynamics on a double well, toy diffusion forward process.
"""
import numpy as np

rng = np.random.default_rng(22)

STATES = ["Sunny", "Cloudy", "Rainy"]
P = np.array(
    [
        [0.7, 0.2, 0.1],  # from Sunny
        [0.3, 0.4, 0.3],  # from Cloudy
        [0.2, 0.4, 0.4],  # from Rainy
    ]
)


def simulate_chain(p, steps, start=0):
    path = [start]
    for _ in range(steps):
        path.append(int(rng.choice(len(p), p=p[path[-1]])))
    return path


print("=" * 60)
print("DEMO 1: Markov chain weather (next depends ONLY on current)")
print("=" * 60)
path = simulate_chain(P, 200_000)
print("Transition matrix rows (Sunny/Cloudy/Rainy):")
print(P)
print("First 10 simulated days:", " -> ".join(STATES[s] for s in path[:10]))
# Markov property check: P(next=Sunny | now=Cloudy) should not care about yesterday.
counts = {}
for prev, cur, nxt in zip(path, path[1:], path[2:]):
    if cur == 1:
        key = prev
        tot, suns = counts.get(key, (0, 0))
        counts[key] = (tot + 1, suns + (nxt == 0))
print("P(tomorrow=Sunny | today=Cloudy), split by what YESTERDAY was:")
probs = []
for prev in range(3):
    tot, suns = counts[prev]
    pr = suns / tot
    probs.append(pr)
    print(f"  yesterday={STATES[prev]:6s}: {pr:.3f}  (matrix says {P[1, 0]:.3f})")
assert all(abs(pr - P[1, 0]) < 0.02 for pr in probs)
print("PASS: all three agree -> the past beyond 'today' adds nothing (Markov).")

print()
print("=" * 60)
print("DEMO 2: 1D random walk, expected distance ~ sqrt(n)")
print("=" * 60)
num_walks = 4000
checkpoints = [10, 100, 1000, 10000]
steps = rng.choice([-1, 1], size=(num_walks, checkpoints[-1]))
positions = np.cumsum(steps, axis=1)
rms = [float(np.sqrt(np.mean(positions[:, n - 1] ** 2.0))) for n in checkpoints]
for n, r in zip(checkpoints, rms):
    print(f"  n={n:6d}: RMS distance = {r:8.2f}   sqrt(n) = {np.sqrt(n):8.2f}")
slope, _ = np.polyfit(np.log(checkpoints), np.log(rms), 1)
print(f"Fitted exponent in distance ~ n^a: a = {slope:.3f} (theory: 0.5)")
assert abs(slope - 0.5) < 0.03
print("PASS: 100x more steps only buys you ~10x more distance.")

print()
print("=" * 60)
print("DEMO 3: Stationary distribution, two independent ways")
print("=" * 60)
pi = np.ones(3) / 3
for _ in range(200):
    pi = pi @ P
freq = np.bincount(path, minlength=3) / len(path)
print(f"Power iteration on P:      {np.round(pi, 4)}")
print(f"Long-run visit frequencies: {np.round(freq, 4)}")
assert np.allclose(pi, pi @ P, atol=1e-10), "stationary means pi @ P == pi"
assert np.allclose(pi, freq, atol=0.01)
print("PASS: both methods agree; pi @ P = pi (the chain's long-run climate).")

print()
print("=" * 60)
print("DEMO 4: Langevin dynamics on a double-well energy U(x)=(x^2-1)^2")
print("=" * 60)


def grad_u(x):
    return 4.0 * x * (x * x - 1.0)


def langevin(temp, n_steps=20000, dt=0.01, x0=0.1):
    x = x0
    samples = []
    for _ in range(n_steps):
        x = x - dt * grad_u(x) + np.sqrt(2.0 * temp * dt) * rng.standard_normal()
        samples.append(x)
    return np.array(samples[n_steps // 4 :])  # drop burn-in


for temp in (0.01, 0.5):
    s = langevin(temp)
    frac_right = float(np.mean(s > 0))
    spread = float(np.std(s))
    print(f"  T={temp:4.2f}: fraction of time in right well = {frac_right:.2f}, std = {spread:.3f}")
    if temp == 0.01:
        assert frac_right > 0.99 or frac_right < 0.01, "low T should stay in one well"
        assert np.all(np.abs(np.abs(s) - 1.0) < 0.5), "low T hugs a minimum at +/-1"
    else:
        assert 0.15 < frac_right < 0.85, "high T should hop between both wells"
print("PASS: T->0 collapses to one minimum; higher T explores both wells.")

print()
print("=" * 60)
print("DEMO 5: Toy diffusion-model forward process (data -> pure noise)")
print("=" * 60)
n_pts = 4000
centers = rng.choice([-2.0, 2.0], size=(n_pts, 2))
x0 = centers + 0.1 * rng.standard_normal((n_pts, 2))  # 2D "data": 4 tight blobs
num_t = 100
betas = np.linspace(1e-3, 0.05, num_t)
alpha_bar = np.cumprod(1.0 - betas)
print("Data: tight blobs at (+/-2, +/-2). Forward: x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps")
print(f"{'t':>4} {'signal-to-noise':>16} {'mean |x|':>10} {'cov diag':>18}")
prev_snr = np.inf
for t_idx in [0, 24, 49, 74, 99]:
    ab = alpha_bar[t_idx]
    x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * rng.standard_normal((n_pts, 2))
    snr = ab / (1 - ab)
    cov = np.cov(x_t.T)
    print(f"{t_idx:4d} {snr:16.3f} {np.mean(np.linalg.norm(x_t, axis=1)):10.3f}"
          f"   [{cov[0, 0]:.2f}, {cov[1, 1]:.2f}]")
    assert snr < prev_snr, "SNR must strictly decrease over time"
    prev_snr = snr
final = np.sqrt(alpha_bar[-1]) * x0 + np.sqrt(1 - alpha_bar[-1]) * rng.standard_normal((n_pts, 2))
assert np.all(np.abs(final.mean(axis=0)) < 0.1), "final mean should be ~0"
assert np.all(np.abs(np.diag(np.cov(final.T)) - (alpha_bar[-1] * 4.0 + 1 - alpha_bar[-1])) < 0.4)
# structure check: at t=0 points cluster near corners, at the end they don't
frac_near_corner_start = np.mean(np.min(np.abs(np.abs(x0) - 2.0), axis=1) < 0.5)
frac_near_corner_end = np.mean(np.min(np.abs(np.abs(final) - 2.0), axis=1) < 0.5)
print(f"Fraction of points hugging a blob center: start {frac_near_corner_start:.2f},"
      f" end {frac_near_corner_end:.2f}")
assert frac_near_corner_start > 0.99
print("PASS: SNR falls monotonically; the blob structure dissolves into gaussian noise.")
print("Reverse process (the learned part of a diffusion model) undoes this")
print("one small step at a time: predict the noise, subtract a little of it, repeat.")

print()
print("ALL STOCHASTIC DEMOS PASSED")
