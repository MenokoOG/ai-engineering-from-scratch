"""Lesson 04 - Calculus for ML. From-scratch derivatives, gradients, and gradient descent."""
import numpy as np


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------- Part 1: numerical derivative (central difference) ----------

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


banner("PART 1: Numerical derivative (central difference)")
tests = [
    ("f(x)=x^2 at x=3", lambda x: x * x, 3.0, 6.0),
    ("f(x)=sin(x) at x=0", np.sin, 0.0, 1.0),
    ("f(x)=e^x at x=1", np.exp, 1.0, np.e),
    ("f(x)=1/x at x=2", lambda x: 1 / x, 2.0, -0.25),
]
for name, f, x, exact in tests:
    approx = derivative(f, x)
    print(f"  {name}: numeric={approx:.6f}  exact={exact:.6f}")
    assert abs(approx - exact) < 1e-6, name
print("  All derivative checks passed.")


# ---------- Part 2: gradient of a multivariable function ----------

def gradient(f, w, h=1e-5):
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)
    for i in range(w.size):
        step = np.zeros_like(w)
        step[i] = h
        grad[i] = (f(w + step) - f(w - step)) / (2 * h)
    return grad


banner("PART 2: Gradient of f(x,y) = x^2 + 3y^2 + x*y")


def f2(w):
    x, y = w
    return x ** 2 + 3 * y ** 2 + x * y


point = np.array([1.0, 2.0])
g_numeric = gradient(f2, point)
g_exact = np.array([2 * 1.0 + 2.0, 6 * 2.0 + 1.0])  # [2x+y, 6y+x]
print(f"  point={point}, numeric grad={g_numeric}, exact grad={g_exact}")
assert np.allclose(g_numeric, g_exact, atol=1e-6)
print("  Gradient matches hand-computed answer.")


# ---------- Part 3: gradient descent on a simple loss ----------

banner("PART 3: Gradient descent, update rule w = w - lr * dL/dw")
# Loss: L(w) = (w - 4)^2, minimum at w = 4.
loss = lambda w: (w - 4.0) ** 2
dloss = lambda w: 2.0 * (w - 4.0)

w = 0.0
lr = 0.1
for step in range(60):
    w = w - lr * dloss(w)
    if step % 15 == 0:
        print(f"  step {step:3d}: w={w:.5f}  loss={loss(w):.6f}")
print(f"  final: w={w:.6f} (true minimum is 4)")
assert abs(w - 4.0) < 1e-4
print("  Converged to the minimum.")


# ---------- Part 4: why step size (learning rate) matters ----------

banner("PART 4: Step size matters (same loss, three learning rates)")


def run_gd(lr, steps=30):
    w = 0.0
    for _ in range(steps):
        w = w - lr * dloss(w)
    return w, loss(w)


for lr_try, label in [(0.01, "too small"), (0.1, "good"), (1.1, "too large")]:
    w_end, l_end = run_gd(lr_try)
    print(f"  lr={lr_try:<5} ({label:9s}): final w={w_end:12.4f}  loss={l_end:.4e}")

w_small, l_small = run_gd(0.01)
w_good, l_good = run_gd(0.1)
w_big, l_big = run_gd(1.1)
assert l_good < l_small, "good lr should beat tiny lr in same steps"
assert l_big > 1.0, "too-large lr should diverge"
print("  Small lr = slow. Good lr = converges. Large lr = blows up.")


# ---------- Part 5: gradient checking (numerical vs analytic) ----------

banner("PART 5: Gradient checking on a tiny linear model")
# Model: pred = w @ x, loss = mean squared error over 5 points.
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
y = rng.normal(size=5)


def mse(w):
    return np.mean((X @ w - y) ** 2)


def mse_grad_analytic(w):
    return 2.0 / len(y) * X.T @ (X @ w - y)


w0 = rng.normal(size=3)
g_a = mse_grad_analytic(w0)
g_n = gradient(mse, w0)
rel_err = np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + np.linalg.norm(g_n))
print(f"  analytic grad = {g_a}")
print(f"  numeric  grad = {g_n}")
print(f"  relative error = {rel_err:.2e}")
assert rel_err < 1e-7
print("  Gradient check passed (analytic matches numeric).")

banner("ALL CHECKS PASSED")
