# Course Feedback — 03 Matrix Transformations

**Reviewed:** code/transformations.py (run, exit 0), docs/en.md, quiz.json (no Julia files in this lesson's code folder)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Determinant as volume scaling factor" block.** The lines are written as `| det(Reflection) | = -1` etc. If the vertical bars mean absolute value (the standard reading), the reflection line is impossible — an absolute value cannot be -1. If they are just decoration, the scale line `| det(Scale sx, sy) | = sx * sy` is fine but the notation is misleading either way. Fix: drop the bars and write `det(Reflection) = -1`, `det(Scale) = sx * sy`.
2. **docs/en.md, Key Terms table ("Shearing matrix ... Determinant is 1").** True only for a single-axis shear. The lesson's own `shearing_2d(kx, ky)` builds `[[1, kx], [ky, 1]]`, whose determinant is `1 - kx*ky`, not 1, when both parameters are nonzero. Fix: say "determinant is 1 for a single-axis shear."

## Nitpicks & suggestions

- The composition demo variable `rotate_then_scale = mat_mul(S, R)` is correct math (R applied first) but the name reads backwards at first glance; a comment would help.
- `eigenvector_2x2` returns a unit vector but the docs' worked example uses unnormalized [1,1] and [1,-1]; the mismatch between printed values (0.7071...) and the doc's integers may confuse.

## What's solid

- All numeric claims verified: rotation of (2,1) by 45 deg → (0.71, 2.12), S@R = [[0,-2],[0.5,0]], R@S = [[0,-0.5],[2,0]], eigenpairs of [[2,1],[1,2]], and det(composition) = product of dets — all correct at runtime.
- The eigenvalue/eigenvector from-scratch code is correct (verified A@v = lambda*v in the output), and all five quiz answers are right.
