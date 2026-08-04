# 19 — Complex Numbers

> A complex number is a 2D arrow, and multiplying by one rotates the plane.

**Project:** A Complex class built from scratch (add, mul, conjugate, magnitude, polar), verified against Python's built-ins, then used for Euler's formula via Taylor series, roots of unity, rotation-by-multiplication, and a mini RoPE demo where attention scores depend only on the gap between token positions. A TypeScript version covers the class, rotations, and roots of unity.

## What I built
- Complex class: add, mul, conjugate, magnitude, polar form, from_polar
- Verified (3+2i)(1+4i) = -5+14i and z * conj(z) = |z|^2 against Python's complex type
- Euler's formula e^(i*theta) = cos + i*sin, computed by summing the Taylor series of e^z
- Roots of unity for N = 3, 4, 8: each root^N returns to 1, and each set sums to 0
- Rotation demo: multiplying by e^(i*90deg) walks a point around the unit circle; angles add
- Mini RoPE: token embeddings rotated by position * theta; positions (5,3), (12,10), (100,98) all produce the identical attention score
- project.ts: the same Complex class with rotation demo and roots of unity

## Main points learned
- i is defined by i*i = -1. A complex number a + bi is just a 2D point (a, b) with extra powers.
- Multiplying complex numbers multiplies their lengths and adds their angles. That is the whole magic.
- e^(i*theta) is the point on the unit circle at angle theta — a pure rotation with length 1.
- The conjugate flips the imaginary part; z times its conjugate gives length squared, always real.
- The N roots of unity are N evenly spaced points on the unit circle, and they cancel to zero by symmetry.
- Rotations compose by adding angles, which turns messy trig into simple multiplication.
- Rotating embedding pairs by position makes dot products depend only on position differences — RoPE in one line.

## The algorithms, explained simply
**Complex multiplication.** Use the school rule (a+bi)(c+di) and replace i*i with -1. Geometrically, it is a "rotate and scale" command: lengths multiply, angles add — like stacking two turntable moves into one.

**Euler's formula by series.** Feed i*theta into the ordinary e^x sum (1 + z + z^2/2! + ...). The powers of i cycle through 1, i, -1, -i, sorting the terms into a cosine pile and a sine pile. Compound growth in an imaginary direction does not grow — it turns.

**Roots of unity.** Ask "which numbers, multiplied by themselves N times, give 1?" The answers are N evenly spaced points around the unit circle, like N people spaced around a round table; their positions are so balanced they average to the exact center, zero.

**Rotation by multiplication.** To rotate any point by angle theta, multiply it by e^(i*theta). No rotation matrix, no sin/cos bookkeeping — one multiply does it, and doing it twice just adds the angles.

**Mini RoPE.** Treat each 2D pair of an embedding as one complex number, and rotate it by (position x theta) — like setting a clock hand forward by the token's position. When two tokens compare via dot product, their absolute clock settings cancel and only the difference between them remains: relative position, encoded for free.

## How this shows up in AI
- RoPE in LLaMA, Qwen, and most modern LLMs rotates query/key pairs exactly this way, so attention naturally sees relative distance between tokens.
- Different embedding pairs get different theta values (fast and slow clock hands), giving the model short-range and long-range position sense at once.
- Roots of unity are the frequencies of the FFT, which powers fast convolutions and signal processing in audio models.
- The rotate-don't-stretch property (magnitude preserved) is why RoPE encodes position without distorting what the token means.

## Run it
```
cd 19-complex-numbers
python3 project.py
npx tsx project.ts
```
