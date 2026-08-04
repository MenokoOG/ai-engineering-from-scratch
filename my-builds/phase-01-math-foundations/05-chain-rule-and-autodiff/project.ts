// Lesson 05 - TypeScript port of the core Value autograd engine. No dependencies.
// Run: npx tsx project.ts

class Value {
  data: number;
  grad = 0;
  private prev: Value[];
  private backwardFn: () => void = () => {};

  constructor(data: number, children: Value[] = []) {
    this.data = data;
    this.prev = children;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o]);
    out.backwardFn = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o]);
    out.backwardFn = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out.backwardFn = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.prev) build(child);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i].backwardFn();
  }
}

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) < tol;
}

console.log("Demo 1: y = tanh(a*b + c) with a=2, b=-1, c=1");
const a = new Value(2);
const b = new Value(-1);
const c = new Value(1);
const y = a.mul(b).add(c).tanh();
y.backward();
const t = Math.tanh(-1);
console.log(`  y = ${y.data.toFixed(6)} (expected ${t.toFixed(6)})`);
console.log(`  dy/da = ${a.grad.toFixed(6)} (expected ${((1 - t * t) * -1).toFixed(6)})`);
console.log(`  dy/db = ${b.grad.toFixed(6)} (expected ${((1 - t * t) * 2).toFixed(6)})`);
console.log(`  dy/dc = ${c.grad.toFixed(6)} (expected ${(1 - t * t).toFixed(6)})`);
if (!close(y.data, t) || !close(a.grad, (1 - t * t) * -1) ||
    !close(b.grad, (1 - t * t) * 2) || !close(c.grad, 1 - t * t)) {
  throw new Error("gradient mismatch");
}

console.log("Demo 2: gradient accumulation, y = x*x with x=3");
const x = new Value(3);
const y2 = x.mul(x);
y2.backward();
console.log(`  dy/dx = ${x.grad} (correct is 2x = 6; += sums both paths)`);
if (x.grad !== 6) throw new Error("accumulation broken");

console.log("Demo 3: gradient check vs numerical differencing");
const f = (v: number) => {
  const p = new Value(v);
  return p.mul(p).add(p.mul(4)).tanh(); // tanh(v^2 + 4v)
};
const p = new Value(0.3);
const out = p.mul(p).add(p.mul(4)).tanh();
out.backward();
const h = 1e-6;
const numeric = (f(0.3 + h).data - f(0.3 - h).data) / (2 * h);
console.log(`  engine  = ${p.grad.toFixed(8)}`);
console.log(`  numeric = ${numeric.toFixed(8)}`);
if (!close(p.grad, numeric, 1e-6)) throw new Error("gradient check failed");

console.log("ALL CHECKS PASSED");
