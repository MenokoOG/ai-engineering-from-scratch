// Lesson 19 (TypeScript): a Complex number class — add, mul, magnitude,
// rotation by multiplication, and roots of unity.
// Run: npx tsx project.ts

class Complex {
  constructor(
    public re: number,
    public im: number = 0,
  ) {}

  add(other: Complex): Complex {
    return new Complex(this.re + other.re, this.im + other.im);
  }

  mul(other: Complex): Complex {
    return new Complex(
      this.re * other.re - this.im * other.im,
      this.re * other.im + this.im * other.re,
    );
  }

  conjugate(): Complex {
    return new Complex(this.re, -this.im);
  }

  magnitude(): number {
    return Math.sqrt(this.re * this.re + this.im * this.im);
  }

  static fromPolar(r: number, theta: number): Complex {
    return new Complex(r * Math.cos(theta), r * Math.sin(theta));
  }

  isClose(other: Complex, tol = 1e-9): boolean {
    return Math.abs(this.re - other.re) < tol && Math.abs(this.im - other.im) < tol;
  }

  toString(): string {
    const sign = this.im >= 0 ? "+" : "-";
    return `(${this.re.toPrecision(4)} ${sign} ${Math.abs(this.im).toPrecision(4)}i)`;
  }
}

function check(cond: boolean, label: string): void {
  if (!cond) throw new Error(`FAIL: ${label}`);
  console.log(`  ok: ${label}`);
}

console.log("=== 1) basics: (3+2i)(1+4i) should be -5+14i ===");
const a = new Complex(3, 2);
const b = new Complex(1, 4);
console.log(`  a = ${a}, b = ${b}`);
console.log(`  a + b = ${a.add(b)}`);
console.log(`  a * b = ${a.mul(b)}`);
check(a.mul(b).isClose(new Complex(-5, 14)), "(3+2i)(1+4i) = -5+14i");
check(Math.abs(a.magnitude() - Math.sqrt(13)) < 1e-12, "|3+2i| = sqrt(13)");
check(a.mul(a.conjugate()).isClose(new Complex(13, 0)), "z * conj(z) = |z|^2");

console.log("\n=== 2) rotation by multiplication ===");
const rot90 = Complex.fromPolar(1, Math.PI / 2);
let p = new Complex(1, 0);
const path: string[] = [p.toString()];
for (let i = 0; i < 4; i++) {
  p = p.mul(rot90);
  path.push(p.toString());
}
console.log(`  (1+0i) times e^(i*90deg), four times: ${path.join(" -> ")}`);
check(p.isClose(new Complex(1, 0)), "four 90-degree turns return home");
const v = new Complex(3, 4);
const rotated = v.mul(Complex.fromPolar(1, 0.7));
check(Math.abs(rotated.magnitude() - v.magnitude()) < 1e-12, "rotation keeps length: |z*e^(i*t)| = |z|");
const twoSteps = Complex.fromPolar(1, 0.3).mul(Complex.fromPolar(1, 0.5));
check(twoSteps.isClose(Complex.fromPolar(1, 0.8)), "rotations compose: angles add (0.3+0.5=0.8)");

console.log("\n=== 3) roots of unity ===");
for (const N of [3, 4, 8]) {
  const roots: Complex[] = [];
  for (let k = 0; k < N; k++) {
    roots.push(Complex.fromPolar(1, (2 * Math.PI * k) / N));
  }
  const sum = roots.reduce((s, r) => s.add(r), new Complex(0, 0));
  console.log(`  N=${N}: sum of roots = ${sum} (magnitude ${sum.magnitude().toExponential(2)})`);
  check(sum.magnitude() < 1e-9, `N=${N} roots sum to 0`);
  for (const r of roots) {
    let z = new Complex(1, 0);
    for (let i = 0; i < N; i++) z = z.mul(r);
    if (!z.isClose(new Complex(1, 0))) throw new Error(`FAIL: root^${N} != 1`);
  }
  console.log(`  ok: every N=${N} root raised to the ${N}th power returns to 1`);
}

console.log("\nAll TypeScript checks passed.");
