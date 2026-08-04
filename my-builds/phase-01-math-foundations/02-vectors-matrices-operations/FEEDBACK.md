# Course Feedback — 02 Vectors, Matrices & Operations

**Reviewed:** code/matrices.py (run, exit 0), docs/en.md, quiz.json (no Julia files in this lesson's code folder)
**Verdict:** Clean

## Bugs & errors

None found. All arithmetic in the docs checks out (A@B = [[19,22],[43,50]], det([[4,7],[2,6]]) = 10, inverse verified as A@A^-1 = I, broadcasting example correct). The code's cofactor determinant, 2x2 inverse, and broadcast add are all correct and verified at runtime.

## Nitpicks & suggestions

- Header says "Languages: Python, Julia" but the lesson contains no Julia code at all. Either drop Julia from the header or add the snippet.
- The docs' simpler `Matrix.__add__` (Step 2) has no shape check and no broadcasting; the shipped matrices.py version does both. Students who type in the docs version get confusing zip-style errors on mismatched shapes. A one-line note about the difference would help.
- `inverse_2x2` in the docs uses `det == 0` exact float comparison; the .py file correctly uses `abs(det) < 1e-10`. The docs version can miss numerically singular matrices.

## What's solid

- The from-scratch layer demo (`relu(W @ x + b)`) works end to end and shapes are explained at every step.
- Quiz answers and explanations are all correct, and the element-wise vs matmul distinction is well drilled.
