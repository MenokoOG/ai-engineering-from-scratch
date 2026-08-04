"""Lesson 20: Fourier Transform from scratch.
DFT, radix-2 FFT, convolution theorem, zero-padding, sinusoidal positional encodings.
"""
import cmath
import math
import time

import numpy as np

rng = np.random.default_rng(20)


def dft(x):
    """Plain O(n^2) DFT with explicit loops."""
    n = len(x)
    out = []
    for k in range(n):
        s = 0j
        for t in range(n):
            s += x[t] * cmath.exp(-2j * math.pi * k * t / n)
        out.append(s)
    return out


def fft(x):
    """Recursive radix-2 FFT. len(x) must be a power of 2."""
    n = len(x)
    if n == 1:
        return [complex(x[0])]
    evens = fft(x[0::2])
    odds = fft(x[1::2])
    out = [0j] * n
    for k in range(n // 2):
        tw = cmath.exp(-2j * math.pi * k / n) * odds[k]
        out[k] = evens[k] + tw
        out[k + n // 2] = evens[k] - tw
    return out


def ifft(x):
    n = len(x)
    conj = [v.conjugate() for v in x]
    return [v.conjugate() / n for v in fft(conj)]


def circular_convolve(a, b):
    """Direct O(n^2) circular convolution."""
    n = len(a)
    out = []
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += a[j] * b[(k - j) % n]
        out.append(s)
    return out


def positional_encoding(num_pos, d_model):
    """Transformer sin/cos positional encodings with geometric wavelengths."""
    pe = np.zeros((num_pos, d_model))
    for pos in range(num_pos):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos, i] = math.sin(angle)
            pe[pos, i + 1] = math.cos(angle)
    return pe


def cosine_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("=" * 60)
print("DEMO 1: DFT finds the frequencies in a 2-tone signal")
print("=" * 60)
fs = 64
n = 64
t = np.arange(n) / fs
signal = 1.0 * np.sin(2 * math.pi * 5 * t) + 0.5 * np.sin(2 * math.pi * 12 * t)
spectrum = dft(list(signal))
mags = [abs(v) for v in spectrum[: n // 2]]
peaks = sorted(range(len(mags)), key=lambda k: -mags[k])[:2]
print(f"Signal = sin(2pi*5t) + 0.5*sin(2pi*12t), {n} samples at {fs} Hz")
print(f"Top two DFT peaks at bins: {sorted(peaks)} (expected [5, 12])")
assert sorted(peaks) == [5, 12]
assert np.allclose(spectrum, np.fft.fft(signal), atol=1e-8)
print("PASS: DFT matches numpy.fft and recovers both tones.")

print()
print("=" * 60)
print("DEMO 2: Recursive radix-2 FFT, verified + timed vs O(n^2) DFT")
print("=" * 60)
x = list(rng.standard_normal(1024))
t0 = time.perf_counter()
slow = dft(x)
t_dft = time.perf_counter() - t0
t0 = time.perf_counter()
fast = fft(x)
t_fft = time.perf_counter() - t0
assert np.allclose(fast, np.fft.fft(x), atol=1e-6)
assert np.allclose(slow, fast, atol=1e-6)
assert np.allclose(ifft(fast), x, atol=1e-8)
print(f"n=1024: DFT loops {t_dft*1000:8.1f} ms   recursive FFT {t_fft*1000:6.1f} ms")
print(f"Speedup: {t_dft / t_fft:.0f}x  (DFT is O(n^2), FFT is O(n log n))")
print("PASS: FFT matches numpy.fft.fft; ifft(fft(x)) == x.")

print()
print("=" * 60)
print("DEMO 3: Convolution theorem (time conv = freq multiply)")
print("=" * 60)
a = list(rng.standard_normal(64))
b = list(rng.standard_normal(64))
direct = circular_convolve(a, b)
via_fft = [v.real for v in ifft([p * q for p, q in zip(fft(a), fft(b))])]
err = max(abs(u - v) for u, v in zip(direct, via_fft))
print(f"Direct circular convolution vs ifft(fft(a)*fft(b)): max error {err:.2e}")
assert err < 1e-9
pad = 64
lin_via_fft = [
    v.real
    for v in ifft([p * q for p, q in zip(fft(a + [0.0] * pad), fft(b + [0.0] * pad))])
][: 2 * 64 - 1]
assert np.allclose(lin_via_fft, np.convolve(a, b), atol=1e-8)
print("PASS: zero-padded FFT product also matches np.convolve (linear conv).")

print()
print("=" * 60)
print("DEMO 4: Zero-padding interpolates but adds no true resolution")
print("=" * 60)


def count_peaks(mags, thresh_frac=0.2):
    top = max(mags)
    c = 0
    for i in range(1, len(mags) - 1):
        if mags[i] > mags[i - 1] and mags[i] >= mags[i + 1] and mags[i] > thresh_frac * top:
            c += 1
    return c


fs = 32
n_short = 32
t_short = np.arange(n_short) / fs
two_close = np.sin(2 * math.pi * 10.0 * t_short) + np.sin(2 * math.pi * 10.5 * t_short)
padded = list(two_close) + [0.0] * (256 - n_short)
mag_pad = [abs(v) for v in fft(padded)[:128]]
peaks_padded = count_peaks(mag_pad)

n_long = 256
t_long = np.arange(n_long) / fs
two_close_long = np.sin(2 * math.pi * 10.0 * t_long) + np.sin(2 * math.pi * 10.5 * t_long)
mag_long = [abs(v) for v in fft(list(two_close_long))[:128]]
peaks_long = count_peaks(mag_long)

print("Two tones 0.5 Hz apart (10.0 and 10.5 Hz), sampled at 32 Hz.")
print(f"  1s of data zero-padded 32->256 samples: {peaks_padded} peak(s) visible")
print(f"  8s of real data (256 samples):          {peaks_long} peak(s) visible")
assert peaks_padded == 1 and peaks_long == 2
single = np.sin(2 * math.pi * 10.25 * t_short)
mag1 = [abs(v) for v in fft(list(single) + [0.0] * (512 - n_short))]
grid_hz = fs / 512
est = max(range(256), key=lambda k: mag1[k]) * grid_hz
print(f"  Bonus: padding a single 10.25 Hz tone locates its peak at {est:.2f} Hz")
print("         (finer grid = interpolation; separating close tones needs more data)")
assert abs(est - 10.25) < 0.1
print("PASS: padding sharpened the grid but could not split the two tones.")

print()
print("=" * 60)
print("DEMO 5: Sinusoidal positional encodings (Transformer scheme)")
print("=" * 60)
num_pos, d_model = 50, 16
pe = positional_encoding(num_pos, d_model)
dists = np.linalg.norm(pe[:, None, :] - pe[None, :, :], axis=-1)
min_offdiag = dists[~np.eye(num_pos, dtype=bool)].min()
print(f"{num_pos} positions, d_model={d_model}, wavelengths 2pi..2pi*10000 (geometric)")
print(f"Smallest distance between two different positions: {min_offdiag:.3f} (> 0 => unique)")
assert min_offdiag > 1e-3
for p in (10, 25):
    s1 = cosine_sim(pe[p], pe[p + 1])
    s5 = cosine_sim(pe[p], pe[p + 5])
    s20 = cosine_sim(pe[p], pe[p + 20])
    print(f"cos-sim from pos {p}:  +1 -> {s1:.3f}   +5 -> {s5:.3f}   +20 -> {s20:.3f}")
    assert s1 > s5 > s20
print("PASS: every position unique; nearby positions look similar, far ones don't.")

print()
print("ALL FOURIER DEMOS PASSED")
