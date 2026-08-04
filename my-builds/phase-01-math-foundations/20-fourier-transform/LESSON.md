# 20 — Fourier Transform

> Any signal is a recipe of sine waves; the Fourier transform reads out the recipe.

**Project:** Built the DFT with plain loops, a recursive FFT verified against numpy with a timing race, a numeric proof of the convolution theorem, a zero-padding demo, and the Transformer's sin/cos positional encodings.

## What I built
- O(n^2) DFT with two plain loops. It finds the two tones (5 Hz and 12 Hz) hidden in a signal.
- Recursive radix-2 FFT. Matches `numpy.fft.fft` and beats my DFT by ~700x at n=1024.
- Convolution theorem demo. Convolving in time equals multiplying spectra, checked numerically.
- Zero-padding demo. Padding gives a finer frequency grid but cannot split two close tones.
- Sinusoidal positional encodings. Every position gets a unique code; neighbors get similar codes.

## Main points learned
- The Fourier transform converts "signal over time" into "how much of each frequency."
- The DFT is just dot products: compare the signal against every spinning frequency.
- FFT gets the same answer in O(n log n) by splitting into even and odd samples, recursively.
- Convolution (an O(n^2) job) becomes cheap multiplication in frequency space.
- Zero-padding interpolates the spectrum. Only more real data adds true resolution.
- Frequency resolution = 1 / (recording length). Record longer to tell close tones apart.
- Transformers use sin/cos waves at geometric wavelengths to encode word position.

## The algorithms, explained simply
**DFT.** Imagine testing a smoothie for each fruit by comparing it against every pure fruit flavor one at a time. The DFT compares the signal against every pure frequency and scores the match. Every score needs a full pass over the signal, so it is slow: n frequencies times n samples.

**FFT.** Split the samples into evens and odds, solve each half, then merge with a twist (a butterfly step). Like sorting mail by splitting the pile in half again and again instead of scanning the whole pile for each slot. Same answer, hugely fewer steps.

**Convolution theorem.** Convolution means sliding one signal over another and summing overlaps, like smearing paint with a stencil. In frequency space that whole slide-and-sum collapses to plain multiplication, one frequency at a time. So fast filtering = FFT, multiply, inverse FFT.

**Zero-padding.** Adding silence to the end of a recording gives the spectrum more sample points, like printing the same photo at a higher DPI. The picture looks smoother but contains no new detail. Two tones closer than 1/(recording length) still blur into one lobe.

**Sinusoidal positional encoding.** Give each position a set of clock hands spinning at fast-to-slow speeds (geometric wavelengths). Fast hands separate neighbors; slow hands separate far-apart positions. Read all hands together and every position has a unique, smoothly changing fingerprint.

## How this shows up in AI
- Transformers inject word order with exactly these sin/cos positional encodings.
- Convolution layers in CNNs can be computed via FFT; frameworks use this for large kernels.
- Speech and audio models eat spectrograms, which are windowed FFTs of the waveform.
- Fourier features and RoPE-style rotations reuse the same "spinning frequencies" trick for embeddings.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/20-fourier-transform
python3 project.py
```
