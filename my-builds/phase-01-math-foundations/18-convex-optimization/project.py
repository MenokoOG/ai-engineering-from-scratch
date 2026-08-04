"""Lesson 18 - Convex optimization. Convexity test, Newton vs GD, projected GD, and KKT slackness."""
import numpy as np


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------- Part 1: convexity checker via the midpoint test ----------

def is_convex_midpoint(f, lo=-5.0, hi=5.0, trials=2000, seed=0, tol=1e-9):
    """Convex means: chord midpoint is never below the function midpoint.
    f((a+b)/2) <= (f(a)+f(b))/2 for all a, b."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        a, b = rng.uniform(lo, hi, size=2)
        if f((a + b) / 2) > (f(a) + f(b)) / 2 + tol:
            return False
    return True


banner("PART 1: Convexity checker (midpoint test)")
candidates = [
    ("x^2", lambda x: x ** 2, True),
    ("|x|", abs, True),
    ("e^x", np.exp, True),
    ("5x^2+3x+1", lambda x: 5 * x ** 2 + 3 * x + 1, True),
    ("sin(x)", np.sin, False),
    ("-x^2", lambda x: -x ** 2, False),
    ("x^3", lambda x: x ** 3, False),
]
for name, f, expected in candidates:
    got = is_convex_midpoint(f)
    print(f"  {name:<12} convex? {got}  (expected {expected})")
    assert got == expected, name
print("  Midpoint test: a chord never dips below the curve on a convex function.")


# ---------- Part 2: Newton's method vs gradient descent ----------

banner("PART 2: Newton's method on f(x) = 5x^2 + 3x + 1 (min at x = -0.3)")
f = lambda x: 5 * x ** 2 + 3 * x + 1
df = lambda x: 10 * x + 3
d2f = lambda x: 10.0  # constant curvature

x_newton = 4.0
x_newton = x_newton - df(x_newton) / d2f(x_newton)  # ONE Newton step
print(f"  Newton from x=4: after 1 step x = {x_newton:.10f} (exact min is -0.3)")
assert abs(x_newton - (-0.3)) < 1e-12
print("  Newton lands EXACTLY on the minimum in 1 step (quadratic + true curvature).")

x_gd = 4.0
lr = 0.05  # note: lr = 1/curvature = 0.1 would secretly BE a Newton step here
gd_steps = 0
while abs(x_gd - (-0.3)) > 1e-10:
    x_gd = x_gd - lr * df(x_gd)
    gd_steps += 1
    assert gd_steps < 10000
print(f"  GD (lr=0.05) from x=4: needs {gd_steps} steps to reach the same accuracy.")
assert gd_steps > 5
print("  Newton uses curvature to size the step perfectly; GD guesses with lr.")


# ---------- Part 3: projected gradient descent ----------

banner("PART 3: Projected GD: minimize (x-3)^2 + (y-3)^2 inside the unit disk")
# Unconstrained min is (3,3), outside the disk x^2+y^2 <= 1.
# Constrained answer: closest disk point to (3,3) = (1/sqrt2, 1/sqrt2).


def project_disk(w):
    n = np.linalg.norm(w)
    return w if n <= 1.0 else w / n


target = np.array([3.0, 3.0])
loss = lambda w: np.sum((w - target) ** 2)
grad = lambda w: 2 * (w - target)

w = np.array([-0.5, 0.8])
for step in range(200):
    w = project_disk(w - 0.1 * grad(w))  # gradient step, then snap back inside

expected = np.array([1.0, 1.0]) / np.sqrt(2)
print(f"  solution = {w}")
print(f"  expected = {expected} (edge of disk, toward the target)")
print(f"  constraint x^2+y^2 = {np.sum(w**2):.6f} (must be <= 1)")
assert np.allclose(w, expected, atol=1e-6)
assert np.sum(w ** 2) <= 1.0 + 1e-9
print("  Step downhill, then project back into the allowed set. Repeat.")


# ---------- Part 4: KKT complementary slackness ----------

banner("PART 4: KKT complementary slackness, minimize (x-c)^2 s.t. x <= 1")
# Constraint g(x) = x - 1 <= 0. KKT says lambda * g(x*) = 0:
# either the constraint is inactive (lambda = 0) or tight (g = 0, lambda >= 0).


def solve_constrained(c):
    if c <= 1.0:
        x_star = c            # constraint inactive: unconstrained min is allowed
        lam = 0.0
    else:
        x_star = 1.0          # constraint active: pinned at the wall
        lam = 2 * (c - 1.0)   # stationarity: 2(x-c) + lambda = 0 at x=1
    return x_star, lam


for c, story in [(0.5, "min at 0.5, inside the allowed region"),
                 (3.0, "min at 3.0, blocked by the wall x<=1")]:
    x_star, lam = solve_constrained(c)
    g = x_star - 1.0
    print(f"  c={c}: {story}")
    print(f"    x* = {x_star}, lambda = {lam}, g(x*) = {g:.1f}, lambda*g = {lam * g:.1f}")
    assert abs(lam * g) < 1e-12, "complementary slackness must hold"
    # stationarity check: gradient of Lagrangian is zero at x*
    assert abs(2 * (x_star - c) + lam) < 1e-12

x_in, lam_in = solve_constrained(0.5)
x_on, lam_on = solve_constrained(3.0)
assert lam_in == 0.0 and x_in == 0.5, "inactive constraint => lambda = 0"
assert lam_on > 0.0 and x_on == 1.0, "active constraint => lambda > 0, solution on the wall"
print("  Slackness in one line: you only pay for a wall you are leaning on.")
print("  lambda = 0 when the wall is not touched; lambda > 0 when pressed against it.")


banner("SUMMARY: why SGD still works in non-convex deep learning")
print("  Convex means one global valley and every downhill path reaches it.")
print("  Deep net losses are NOT convex, yet SGD works anyway because:")
print("  - in huge dimensions most flat points are saddles, and noise slides off them;")
print("  - overparameterized nets have many minima that are all good enough;")
print("  - we do not need THE minimum, just a low-loss point that generalizes.")

banner("ALL CHECKS PASSED")
