"""Lesson 13: Numerical stability. Floats lie a little; good algorithms keep the lies small."""
import math
import numpy as np

np.random.seed(13)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) 0.1 + 0.2 != 0.3 and machine epsilon")

a = 0.1 + 0.2
print(f"0.1 + 0.2            = {a!r}")
print(f"0.1 + 0.2 == 0.3     -> {a == 0.3}")
print(f"difference           = {a - 0.3:.2e}")
print(f"math.isclose fixes it-> {math.isclose(a, 0.3)}")
assert a != 0.3
assert math.isclose(a, 0.3)

eps = 1.0
while 1.0 + eps / 2 > 1.0:
    eps /= 2
print(f"discovered machine epsilon (float64) = {eps:.3e}")
print(f"numpy says                           = {np.finfo(np.float64).eps:.3e}")
assert eps == np.finfo(np.float64).eps

# ---------------------------------------------------------------
section("2) float32 range and precision probe")

f32 = np.finfo(np.float32)
print(f"largest float32          = {f32.max:.3e}")
print(f"smallest normal float32  = {f32.tiny:.3e}")
print(f"float32 epsilon          = {f32.eps:.3e}  (~7 decimal digits)")

big = np.float32(1e10)
print(f"1e10 + 1 in float32      = {big + np.float32(1.0):.10e}  (the +1 vanished)")
assert big + np.float32(1.0) == big

with np.errstate(over="ignore"):
    overflow = np.float32(f32.max) * np.float32(2.0)
print(f"max * 2 in float32       = {overflow}  (overflow -> inf)")
assert np.isinf(overflow)

spacing_at_1 = np.spacing(np.float32(1.0))
spacing_at_1e8 = np.spacing(np.float32(1e8))
print(f"gap between neighbors near 1.0  = {spacing_at_1:.3e}")
print(f"gap between neighbors near 1e8  = {spacing_at_1e8:.3e}")
print("bigger numbers -> bigger gaps: precision is relative, not absolute")

# ---------------------------------------------------------------
section("3) naive vs stable softmax")


def softmax_naive(x):
    e = np.exp(x)
    return e / e.sum()


def softmax_stable(x):
    e = np.exp(x - x.max())
    return e / e.sum()


logits = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = softmax_naive(logits)
stable = softmax_stable(logits)
print(f"logits        = {logits}")
print(f"naive softmax = {naive}  (exp(1000) overflowed)")
print(f"stable softmax= {np.round(stable, 6)}")
assert np.isnan(naive).any()
assert not np.isnan(stable).any()
assert math.isclose(stable.sum(), 1.0)

small_logits = np.array([1.0, 2.0, 3.0])
assert np.allclose(softmax_naive(small_logits), softmax_stable(small_logits))
print("on safe inputs, both versions agree exactly")

# ---------------------------------------------------------------
section("4) log-sum-exp trick")


def logsumexp_naive(x):
    return np.log(np.sum(np.exp(x)))


def logsumexp_stable(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


x = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore"):
    naive_lse = logsumexp_naive(x)
stable_lse = logsumexp_stable(x)
print(f"naive  log-sum-exp of {x} = {naive_lse}")
print(f"stable log-sum-exp        = {stable_lse:.6f}")
expected = 1002.0 + math.log(math.exp(-2) + math.exp(-1) + 1.0)
assert np.isinf(naive_lse)
assert math.isclose(stable_lse, expected)
print(f"hand-checked answer       = {expected:.6f}")

# ---------------------------------------------------------------
section("5) catastrophic cancellation")

x = 1e8
bad = math.sqrt(x + 1) - math.sqrt(x)
good = 1.0 / (math.sqrt(x + 1) + math.sqrt(x))  # same value, rewritten
print(f"sqrt(x+1) - sqrt(x) at x=1e8")
print(f"  naive subtraction : {bad:.15e}")
print(f"  rewritten formula : {good:.15e}")
rel_err = abs(bad - good) / good
print(f"  naive relative error ~ {rel_err:.1e} (lost ~8 of 16 digits)")
assert rel_err > 1e-9

y = 1e-12
print(f"(1 + y) - 1 for y=1e-12 gives {(1.0 + y) - 1.0:.6e} (should be 1e-12)")
print("subtracting nearly equal numbers wipes out the leading digits")

# ---------------------------------------------------------------
section("6) finite-difference step size: the error U-shape")


def f(t):
    return math.sin(t)


x0 = 1.0
true_deriv = math.cos(x0)
print(f"f=sin, x=1, true derivative = cos(1) = {true_deriv:.12f}")
print(f"{'h':>10} | {'central diff error':>18}")
errors = {}
for k in range(1, 13):
    h = 10.0 ** (-k)
    approx = (f(x0 + h) - f(x0 - h)) / (2 * h)
    err = abs(approx - true_deriv)
    errors[h] = err
    print(f"{h:>10.0e} | {err:>18.3e}")
best_h = min(errors, key=errors.get)
print(f"best h ~ {best_h:.0e}")
print("big h -> truncation error (formula too crude)")
print("tiny h -> rounding error (subtracting nearly equal floats)")
assert errors[1e-1] > errors[best_h]
assert errors[1e-12] > errors[best_h]
assert 1e-7 <= best_h <= 1e-4

# ---------------------------------------------------------------
section("7) float16 vs bfloat16: range vs precision")


def to_bfloat16(x):
    """Simulate bfloat16: keep float32's sign+exponent+top 7 mantissa bits."""
    as_int = np.float32(x).view(np.uint32)
    truncated = as_int & np.uint32(0xFFFF0000)
    return truncated.view(np.float32)


print(f"{'value':>12} | {'float16':>14} | {'bfloat16 (sim)':>14}")
with np.errstate(over="ignore"):
    for v in [1.0, 1.001, 3.14159265, 70000.0, 1e10, 1e-10, 6.55e4]:
        h = np.float32(np.float16(v))
        b = to_bfloat16(v)
        print(f"{v:>12g} | {h:>14g} | {b:>14g}")

f16 = np.finfo(np.float16)
print(f"\nfloat16  max = {f16.max:g}, eps = {f16.eps:.2e}  (10 mantissa bits: precise, tiny range)")
print(f"bfloat16 max ~ 3.4e38,   eps ~ 7.8e-3  (7 mantissa bits: coarse, float32 range)")
with np.errstate(over="ignore"):
    assert np.isinf(np.float16(70000.0))
assert not np.isinf(to_bfloat16(70000.0))
assert np.float16(1.001) != np.float16(1.0)
assert to_bfloat16(1.001) == to_bfloat16(1.0)
print("70000 overflows float16 but not bfloat16")
print("1.001 survives float16 but rounds to 1.0 in bfloat16")
print("training cares more about range (no inf/nan blowups) than precision -> bf16 wins")

print("\nAll numerical stability checks passed.")
