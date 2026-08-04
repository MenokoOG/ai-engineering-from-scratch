# Course Feedback — 20-fourier-transform

**Reviewed:** code/fourier.py (ran with python3, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

None found (no correctness bugs).

- fourier.py runs cleanly. DFT and FFT agree to 6e-14, FFT -> IFFT reconstructs to 4e-16, FFT-based convolution of [1..5] with [1,1,1] exactly matches direct convolution ([1,3,6,9,12,9,5]), and Parseval's theorem holds to machine precision.
- The windowing demo behaves as claimed: the 7.5 Hz between-bins tone leaks broadly without a window and concentrates around bins 7-8 with Hann/Hamming.
- The DFT/FFT math in docs/en.md (definition, inverse, twiddle-factor butterfly, conjugate symmetry, properties table including the 1/N factor on the multiplication property) all checks out.
- All 5 quiz answers and explanations are correct.

## Nitpicks & suggestions

1. docs/en.md, Step 4 (`convolve_fft` snippet): it calls `idft(Y)` — the O(N^2) inverse — inside the "fast convolution" routine, which silently defeats the FFT speedup the section is teaching. The actual code file correctly uses `ifft`. Suggest changing the doc snippet to `ifft`.
2. docs/en.md, FFT section: "The FFT requires the signal length to be a power of 2." Only radix-2 Cooley-Tukey requires this; mixed-radix and Bluestein FFTs handle any length (numpy's fft does). Worth softening to "this implementation requires...".
3. code/fourier.py edge cases: `fft([])` crashes on `x[0]`, and `hann_window(1)` / `hamming_window(1)` divide by zero (N-1 = 0). Never hit by the demos.
4. code/fourier.py: `write_prompt_output` uses the relative path `outputs/...`, so the output lands wherever you run the script from. complex_numbers.py in lesson 19 does this more robustly with the script's own directory.

## What's solid

- The convolution theorem, Parseval, and FFT-vs-DFT claims are all demonstrated numerically, not just asserted.
- Good coverage of the classic traps: spectral leakage, aliasing, circular vs linear convolution, and the zero-padding "resolution" misconception are all explained correctly.
