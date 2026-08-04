"""Lesson 19: Complex numbers. Numbers with a direction — multiplication rotates."""
import cmath
import math

SEED_NOTE = "deterministic demo: no randomness used"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class Complex:
    def __init__(self, re, im=0.0):
        self.re = float(re)
        self.im = float(im)

    def add(self, other):
        return Complex(self.re + other.re, self.im + other.im)

    def mul(self, other):
        return Complex(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def conjugate(self):
        return Complex(self.re, -self.im)

    def magnitude(self):
        return math.sqrt(self.re * self.re + self.im * self.im)

    def polar(self):
        return self.magnitude(), math.atan2(self.im, self.re)

    @staticmethod
    def from_polar(r, theta):
        return Complex(r * math.cos(theta), r * math.sin(theta))

    def isclose(self, other, tol=1e-9):
        return abs(self.re - other.re) < tol and abs(self.im - other.im) < tol

    def __repr__(self):
        sign = "+" if self.im >= 0 else "-"
        return f"({self.re:.4g} {sign} {abs(self.im):.4g}i)"


# ---------------------------------------------------------------
section("1) Complex class basics and (3+2i)(1+4i) = -5+14i")

a = Complex(3, 2)
b = Complex(1, 4)
prod = a.mul(b)
print(f"a = {a}, b = {b}")
print(f"a + b        = {a.add(b)}")
print(f"a * b        = {prod}")
print(f"conjugate(a) = {a.conjugate()}")
print(f"|a|          = {a.magnitude():.6f}")
r, theta = a.polar()
print(f"a in polar   = length {r:.4f} at angle {theta:.4f} rad")
assert prod.isclose(Complex(-5, 14))
py = complex(3, 2) * complex(1, 4)
assert prod.isclose(Complex(py.real, py.imag))
assert a.isclose(Complex.from_polar(r, theta))
assert math.isclose(a.mul(a.conjugate()).re, a.magnitude() ** 2)
print("verified: matches Python's built-in complex, polar round-trips, z*conj(z)=|z|^2")

# ---------------------------------------------------------------
section("2) Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta)")


def complex_exp(z, terms=40):
    """e^z by Taylor series: 1 + z + z^2/2! + z^3/3! + ..."""
    result = Complex(1, 0)
    term = Complex(1, 0)
    for n in range(1, terms):
        term = term.mul(z).mul(Complex(1.0 / n, 0))
        result = result.add(term)
    return result


for theta in [0.5, math.pi / 3, math.pi, 2.0]:
    series = complex_exp(Complex(0, theta))
    direct = Complex(math.cos(theta), math.sin(theta))
    cm = cmath.exp(1j * theta)
    print(f"theta = {theta:.4f}: series e^(i*theta) = {series},  cos+isin = {direct}")
    assert series.isclose(direct)
    assert series.isclose(Complex(cm.real, cm.imag))
euler_id = complex_exp(Complex(0, math.pi)).add(Complex(1, 0))
print(f"Euler's identity e^(i*pi) + 1 = {euler_id}  (~0)")
assert euler_id.magnitude() < 1e-9

# ---------------------------------------------------------------
section("3) roots of unity: N equally spaced points, sum = 0")

for N in [3, 4, 8]:
    roots = [Complex.from_polar(1.0, 2 * math.pi * k / N) for k in range(N)]
    total = Complex(0, 0)
    for rt in roots:
        total = total.add(rt)
        assert math.isclose(rt.magnitude(), 1.0)
    print(f"N = {N}: roots = {roots if N <= 4 else '(8 points on unit circle)'}")
    print(f"       sum = {total}  (magnitude {total.magnitude():.2e})")
    assert total.magnitude() < 1e-9
    for rt in roots:
        z = Complex(1, 0)
        for _ in range(N):
            z = z.mul(rt)
        assert z.isclose(Complex(1, 0))
print("each root raised to the N-th power returns to 1; the ring of points balances to 0")

# ---------------------------------------------------------------
section("4) rotation by multiplication")

point = Complex(1, 0)
rot90 = Complex.from_polar(1.0, math.pi / 2)
steps = [point]
for _ in range(4):
    steps.append(steps[-1].mul(rot90))
print("start at (1 + 0i), multiply by e^(i*90deg) repeatedly:")
for i, s in enumerate(steps):
    print(f"  after {i} turns: {s}")
assert steps[1].isclose(Complex(0, 1))
assert steps[2].isclose(Complex(-1, 0))
assert steps[4].isclose(point)
rot_a = Complex.from_polar(1.0, 0.3)
rot_b = Complex.from_polar(1.0, 0.5)
combined = rot_a.mul(rot_b)
assert combined.isclose(Complex.from_polar(1.0, 0.8))
print("two rotations multiply into one: angles 0.3 + 0.5 = 0.8 rad. angles add!")
v = Complex(3, 4)
assert math.isclose(v.mul(rot_a).magnitude(), v.magnitude())
print("rotation never changes length: |z * e^(i*theta)| == |z|")

# ---------------------------------------------------------------
section("5) mini RoPE: rotate embeddings by position, get relative encoding")

theta_rope = 0.15                      # rotation per position for this 2D pair
q_vec = Complex(1.0, 0.5)              # a query's 2D embedding pair as a complex number
k_vec = Complex(0.8, -0.3)             # a key's 2D embedding pair


def rope_encode(vec, position, theta):
    return vec.mul(Complex.from_polar(1.0, position * theta))


def attention_score(q_rot, k_rot):
    """2D dot product of the rotated pairs = Re(q * conj(k))."""
    return q_rot.mul(k_rot.conjugate()).re


print(f"query pair {q_vec}, key pair {k_vec}, theta = {theta_rope}/position")
print(f"{'q pos':>6} {'k pos':>6} {'offset':>7} {'score':>10}")
scores = {}
for m, n in [(5, 3), (12, 10), (100, 98), (7, 3), (50, 46)]:
    qr = rope_encode(q_vec, m, theta_rope)
    kr = rope_encode(k_vec, n, theta_rope)
    s = attention_score(qr, kr)
    scores.setdefault(m - n, []).append(s)
    print(f"{m:>6} {n:>6} {m - n:>7} {s:>10.6f}")
for offset, vals in scores.items():
    for s in vals:
        assert math.isclose(s, vals[0], abs_tol=1e-9)
    # score depends only on the offset: equals Re(q * conj(k) * e^(i*offset*theta))
    expected = q_vec.mul(k_vec.conjugate()).mul(
        Complex.from_polar(1.0, offset * theta_rope)).re
    assert math.isclose(vals[0], expected, abs_tol=1e-9)
s_gap2 = scores[2][0]
s_gap4 = scores[4][0]
assert not math.isclose(s_gap2, s_gap4)
print("positions (5,3), (12,10), (100,98) all score identically: only the gap matters")
print("different gap (7,3) scores differently: relative position is encoded for free")

print(f"\n({SEED_NOTE})")
print("All complex-numbers checks passed.")
