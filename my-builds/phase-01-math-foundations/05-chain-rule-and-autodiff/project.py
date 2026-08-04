"""Lesson 05 - Chain rule and autodiff. Micrograd-style Value class built from scratch."""
import math
import random


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


# ---------- Part 1: basic ops and chain rule ----------

banner("PART 1: Value class basics (chain rule at work)")
# y = (a*b + c).tanh() with a=2, b=-1, c=1  ->  inner = -1, y = tanh(-1)
a, b, c = Value(2.0), Value(-1.0), Value(1.0)
inner = a * b + c
y = inner.tanh()
y.backward()
t = math.tanh(-1.0)
print(f"  y = tanh(a*b + c) = {y.data:.6f} (expected {t:.6f})")
print(f"  dy/da = {a.grad:.6f} (expected {(1 - t * t) * -1.0:.6f})")  # b * (1-t^2)
print(f"  dy/db = {b.grad:.6f} (expected {(1 - t * t) * 2.0:.6f})")   # a * (1-t^2)
print(f"  dy/dc = {c.grad:.6f} (expected {(1 - t * t):.6f})")
assert abs(y.data - t) < 1e-12
assert abs(a.grad - (1 - t * t) * -1.0) < 1e-12
assert abs(b.grad - (1 - t * t) * 2.0) < 1e-12
assert abs(c.grad - (1 - t * t)) < 1e-12
print("  Chain rule gradients all correct.")

r = Value(-3.0).relu()
r2 = Value(3.0).relu()
assert r.data == 0.0 and r2.data == 3.0
print("  relu(-3)=0, relu(3)=3. ReLU works.")


# ---------- Part 2: WHY gradients accumulate with += ----------

banner("PART 2: Why _backward uses += (a value used twice)")
# y = x*x. dy/dx = 2x. x is used TWICE, so gradient must add up both paths.
x = Value(3.0)
y = x * x
y.backward()
print(f"  y = x*x with x=3: dy/dx = {x.grad} (correct answer is 2x = 6)")
assert x.grad == 6.0

# Simulate the bug: overwrite instead of accumulate.
x_bug = Value(3.0)
y_bug = x_bug * x_bug
y_bug.grad = 1.0
x_bug.grad = 0.0
x_bug.grad = x_bug.data * y_bug.grad  # '=' keeps only ONE path: gives 3, not 6
print(f"  With '=' instead of '+=': dy/dx = {x_bug.grad} (WRONG, second path lost)")
assert x_bug.grad == 3.0
print("  += sums the gradient from every path a value feeds into.")


# ---------- Part 3: train a tiny 2-layer net on XOR-ish data ----------

banner("PART 3: Tiny 2-layer net on XOR data, trained with this engine")
random.seed(42)
data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]

HID = 4
w1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(HID)]
b1 = [Value(0.0) for _ in range(HID)]
w2 = [Value(random.uniform(-1, 1)) for _ in range(HID)]
b2 = Value(0.0)
params = [w for row in w1 for w in row] + b1 + w2 + [b2]


def forward(xs):
    hidden = []
    for j in range(HID):
        s = b1[j]
        for i in range(2):
            s = s + w1[j][i] * xs[i]
        hidden.append(s.tanh())
    out = b2
    for j in range(HID):
        out = out + w2[j] * hidden[j]
    return out.tanh()


lr = 0.2
for epoch in range(800):
    loss = Value(0.0)
    for xs, target in data:
        pred = forward(xs)
        diff = pred + Value(-target)
        loss = loss + diff * diff
    for p in params:
        p.grad = 0.0
    loss.backward()
    for p in params:
        p.data -= lr * p.grad
    if epoch % 200 == 0:
        print(f"  epoch {epoch:3d}: loss = {loss.data:.6f}")

print(f"  final loss = {loss.data:.6f}")
for xs, target in data:
    pred = forward(xs).data
    print(f"  input {xs} -> pred {pred:+.3f} (target {target})")
    assert (pred > 0.5) == (target > 0.5), "XOR prediction wrong"
assert loss.data < 0.05
print("  Network learned XOR. Backprop engine works end to end.")


# ---------- Part 4: verify one gradient numerically ----------

banner("PART 4: Gradient check (engine vs numerical differencing)")
p = w1[0][0]


def total_loss():
    s = 0.0
    for xs, target in data:
        s += (forward(xs).data - target) ** 2
    return s


for q in params:
    q.grad = 0.0
loss = Value(0.0)
for xs, target in data:
    diff = forward(xs) + Value(-target)
    loss = loss + diff * diff
loss.backward()
analytic = p.grad

h = 1e-6
orig = p.data
p.data = orig + h
lp = total_loss()
p.data = orig - h
lm = total_loss()
p.data = orig
numeric = (lp - lm) / (2 * h)

print(f"  engine gradient   = {analytic:.8f}")
print(f"  numeric gradient  = {numeric:.8f}")
assert abs(analytic - numeric) < 1e-5
print("  Backprop gradient matches the numerical estimate.")

banner("ALL CHECKS PASSED")
