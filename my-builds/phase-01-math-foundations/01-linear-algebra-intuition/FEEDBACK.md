# Course Feedback — 01 Linear Algebra Intuition

**Reviewed:** code/vectors.py (run, exit 0), docs/en.md, quiz.json (Julia files not run — the docs' Julia snippet was checked statically)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, Step 4 (Julia version).** The snippet uses the `⋅` (dot) operator and calls it directly. In Julia, `⋅` lives in the `LinearAlgebra` standard library. Without `using LinearAlgebra` at the top, the snippet fails with `UndefVarError`. Fix: add `using LinearAlgebra` as the first line.

## Nitpicks & suggestions

- `Vector.normalize()` and `project_onto()` divide by zero on a zero vector. A guard or a one-line note would help.
- `cosine_similarity` can return values slightly outside [-1, 1] from float error. `angle_between` clamps for this — good — but the docs never mention why the clamp is there.
- In the Gram-Schmidt demo, the output vectors are printed as `u1, u2, u3`, the same names as the input vectors in the docs' algorithm description. Slightly confusing, but harmless.

## What's solid

- Every worked example in the docs checks out numerically (projection [3,4]→[3,0], sqrt(13) magnitude, LoRA parameter counts 16M → 131K).
- The code runs clean, the rank/independence row-reduction is correct, and all five quiz answers and explanations are right.
