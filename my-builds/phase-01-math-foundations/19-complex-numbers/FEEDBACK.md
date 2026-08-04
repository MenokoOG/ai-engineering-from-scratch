# Course Feedback — 19-complex-numbers

**Reviewed:** code/complex_numbers.py (ran with python3, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Clean

## Bugs & errors

None found.

- complex_numbers.py runs cleanly. I hand-checked the arithmetic: (3+2i)(1+4i) = -5+14i, (5+2i)/(1-3i) = -0.1+1.7i, z*conj(z) = 13. All match the output.
- Euler's identity comes out to 1.22e-16, rotations preserve magnitude, complex multiplication matches the 2x2 rotation matrix to 0 error, roots of unity sum to ~1e-16, and DFT -> IDFT reconstructs to 7e-15.
- The DFT demo's expected peak heights (amp * N/2 = 16 and 8 for sines) are correct, including the -90 degree phase for a sine.
- The positional-encoding demo matches the Transformer paper formula (freq = 1/10000^(2i/d)).
- All 5 quiz answers and explanations are correct, including the multiplication question (-5+14i).

## Nitpicks & suggestions

1. code/complex_numbers.py: the `Complex` class defines `__eq__` but no `__hash__`, so instances become unhashable. Nothing in the lesson needs hashing, so it never bites here.
2. docs/en.md, "Roots of unity" step: "the roots of unity form an orthogonal basis" — strictly it is the N sampled phasor vectors (DFT basis vectors) that are orthogonal, not the N scalar roots themselves. Loose wording, not wrong in spirit.
3. code: `write_skill_output` writes into the lesson folder relative to the script. Fine normally; it degrades gracefully (prints a message) if the folder is read-only.

## What's solid

- Every property claimed in the text (|e^(i*theta)| = 1, sum of roots = 0, rotation-matrix equivalence, perfect DFT/IDFT reconstruction) is numerically verified in the demo output.
- The ML connections (RoPE, sinusoidal positional encodings) are accurate, not hand-wavy.
