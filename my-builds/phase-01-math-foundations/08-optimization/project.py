"""Lesson 08 - Optimization. GD, momentum, RMSProp, Adam from scratch, plus SGD noise and cosine annealing."""
import math
import numpy as np


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------- test functions ----------

def bowl(w):
    x, y = w
    return x ** 2 + 10 * y ** 2


def bowl_grad(w):
    x, y = w
    return np.array([2 * x, 20 * y])


def rosenbrock(w):
    x, y = w
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(w):
    x, y = w
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])


# ---------- optimizers, all sharing one loop shape ----------

def run_gd(grad_fn, w0, lr, steps):
    w = w0.copy()
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad_fn(w)
        path.append(w.copy())
    return w, path


def run_momentum(grad_fn, w0, lr, steps, beta=0.9):
    w, v = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        v = beta * v + grad_fn(w)
        w = w - lr * v
        path.append(w.copy())
    return w, path


def run_rmsprop(grad_fn, w0, lr, steps, beta=0.99, eps=1e-8):
    w, s = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        s = beta * s + (1 - beta) * g * g
        w = w - lr * g / (np.sqrt(s) + eps)
        path.append(w.copy())
    return w, path


def run_adam(grad_fn, w0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    w = w0.copy()
    m, v = np.zeros_like(w0), np.zeros_like(w0)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return w, path


def steps_to_converge(path, loss_fn, tol):
    for i, w in enumerate(path):
        if loss_fn(w) < tol:
            return i
    return None


# ---------- Part 1: race on the bowl ----------

banner("PART 1: Optimizer race on bowl f(x,y) = x^2 + 10y^2 (min at origin)")
w0 = np.array([5.0, 3.0])
STEPS = 500
runs = {
    "GD (lr=0.05)": run_gd(bowl_grad, w0, 0.05, STEPS),
    "Momentum (lr=0.05)": run_momentum(bowl_grad, w0, 0.05, STEPS),
    "RMSProp (lr=0.05)": run_rmsprop(bowl_grad, w0, 0.05, STEPS),
    "Adam (lr=0.05)": run_adam(bowl_grad, w0, 0.05, STEPS),
}
print(f"  {'optimizer':<20} {'steps to loss<1e-6':>19} {'final loss':>13}")
for name, (w_end, path) in runs.items():
    n = steps_to_converge(path, bowl, 1e-6)
    print(f"  {name:<20} {str(n):>19} {bowl(w_end):>13.2e}")
    assert bowl(w_end) < 1e-4, f"{name} failed on bowl"
print("  All four optimizers reach the bottom of the bowl.")


# ---------- Part 2: race on Rosenbrock (the hard one) ----------

banner("PART 2: Race on Rosenbrock (curved valley, min at (1,1))")
w0 = np.array([-1.0, 1.0])
STEPS = 5000
runs = {
    "GD (lr=1e-3)": run_gd(rosenbrock_grad, w0, 1e-3, STEPS),
    "Momentum (lr=1e-3)": run_momentum(rosenbrock_grad, w0, 1e-3, STEPS),
    "RMSProp (lr=1e-2)": run_rmsprop(rosenbrock_grad, w0, 1e-2, STEPS),
    "Adam (lr=1e-2)": run_adam(rosenbrock_grad, w0, 1e-2, STEPS),
}
print(f"  {'optimizer':<20} {'steps to loss<1e-3':>19} {'final loss':>13} {'final point':>18}")
for name, (w_end, path) in runs.items():
    n = steps_to_converge(path, rosenbrock, 1e-3)
    pt = f"({w_end[0]:.3f},{w_end[1]:.3f})"
    print(f"  {name:<20} {str(n):>19} {rosenbrock(w_end):>13.2e} {pt:>18}")
adam_end = runs["Adam (lr=1e-2)"][0]
gd_end = runs["GD (lr=1e-3)"][0]
assert rosenbrock(adam_end) < 1e-3, "Adam should solve Rosenbrock"
assert rosenbrock(adam_end) < rosenbrock(gd_end), "Adam should beat plain GD here"
print("  Adaptive methods handle the curved valley far better than plain GD.")


# ---------- Part 3: too-large learning rate diverges ----------

banner("PART 3: Too-large learning rate diverges (bowl, lr=0.11 vs 0.05)")
# On f = x^2 + 10y^2 the steep axis has curvature 20; GD is stable only if lr < 2/20 = 0.1.
w_ok, _ = run_gd(bowl_grad, np.array([5.0, 3.0]), 0.05, 50)
w_bad, path_bad = run_gd(bowl_grad, np.array([5.0, 3.0]), 0.11, 50)
print(f"  lr=0.05: loss after 50 steps = {bowl(w_ok):.2e}  (converging)")
print(f"  lr=0.11: loss after 50 steps = {bowl(w_bad):.2e}  (DIVERGED)")
print("  Stability limit here is lr < 2/curvature = 0.1. Just past it, boom.")
assert bowl(w_ok) < 1e-3
assert bowl(w_bad) > 1e6


# ---------- Part 4: mini-batch SGD noise on linear regression ----------

banner("PART 4: Mini-batch SGD noise (linear regression, true w=[3,-2], b=1)")
rng = np.random.default_rng(1)
N = 200
X = rng.normal(size=(N, 2))
true_w, true_b = np.array([3.0, -2.0]), 1.0
y = X @ true_w + true_b + 0.1 * rng.normal(size=N)


def sgd_linreg(batch_size, lr=0.05, epochs=150):
    w, b = np.zeros(2), 0.0
    losses = []
    for _ in range(epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            bi = idx[start:start + batch_size]
            Xb, yb = X[bi], y[bi]
            err = Xb @ w + b - yb
            w -= lr * 2 * Xb.T @ err / len(bi)
            b -= lr * 2 * err.mean()
            losses.append(np.mean((X @ w + b - y) ** 2))
    return w, b, np.array(losses)


tails = {}
for bs in [200, 32, 4]:
    w_fit, b_fit, losses = sgd_linreg(bs)
    tail = losses[-30:]  # all runs are converged well before the tail
    tails[bs] = tail
    label = "full batch" if bs == N else f"batch={bs}"
    print(f"  {label:<11} final loss={losses[-1]:.4f}  loss wiggle (std of last 30)={tail.std():.2e}  w={np.round(w_fit,2)}")
    assert np.allclose(w_fit, true_w, atol=0.15), f"batch {bs} failed to fit"

assert tails[4].std() > tails[200].std(), "smaller batches should be noisier"
print("  Smaller batches = noisier loss curve, but all still find the true weights.")
print("  That noise helps real networks hop out of bad spots.")


# ---------- Part 5: cosine annealing schedule ----------

banner("PART 5: Cosine annealing learning rate schedule")


def cosine_lr(step, total, lr_max=0.1, lr_min=0.001):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total))


TOTAL = 100
sched = [cosine_lr(s, TOTAL) for s in range(TOTAL + 1)]
print("  lr over 100 steps: " + ", ".join(f"step {s}: {cosine_lr(s, TOTAL):.4f}" for s in [0, 25, 50, 75, 100]))
assert abs(sched[0] - 0.1) < 1e-12 and abs(sched[-1] - 0.001) < 1e-12
assert all(sched[i] >= sched[i + 1] for i in range(TOTAL)), "cosine schedule must decay smoothly"

# GD on the bowl with cosine schedule vs fixed small lr.
w = np.array([5.0, 3.0])
for s in range(TOTAL):
    w = w - cosine_lr(s, TOTAL, lr_max=0.09) * bowl_grad(w)
w_fixed, _ = run_gd(bowl_grad, np.array([5.0, 3.0]), 0.01, TOTAL)
print(f"  bowl after 100 steps: cosine schedule loss={bowl(w):.2e}, fixed lr=0.01 loss={bowl(w_fixed):.2e}")
assert bowl(w) < bowl(w_fixed)
print("  Start big to move fast, end small to settle precisely.")

banner("ALL CHECKS PASSED")
