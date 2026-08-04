# PR #1 — ready to send

Copy-paste everything below as-is. The fixed file is `stochastic.py` in this same folder.

---

## PR title

```
Fix: stationary_distribution returns all zeros in Phase 1, Lesson 22
```

## Commit message

```
fix stationary_distribution sign bug in phase 1 lesson 22
```

## PR description

```
Working through Phase 1, Lesson 22, the stationary distribution demo
kept printing all zeros, and the convergence table never converged --
stuck around 0.55 error no matter how long I ran the chain.

Dug into it. numpy sometimes hands back the eigenvalue-1 eigenvector
with every component negative. The clip line then zeroes the whole
vector out before normalization ever gets a shot, and the function
returns [0, 0, 0].

The fix is smaller than the bug: drop the clip, divide by the sum.
For a stochastic matrix, Perron-Frobenius says the components of that
eigenvector all carry the same sign, so one division fixes the sign
and normalizes in the same move. Bonus: this matches the version
already printed in the lesson's docs/en.md. The docs were right the
whole time -- only the .py drifted.

Proof it works:
- Analytical stationary now reads 0.5455 / 0.1818 / 0.2727, right on
  top of the empirical run (0.5469 / 0.1826 / 0.2705)
- Convergence error drops from 0.0531 to 0.0022 as the chain grows,
  like it should
- Sanity check with a second transition matrix: pi @ P == pi,
  non-negative, sums to 1
- Full script runs clean, exit 0

Solid course. A bug like this is a good pressure point -- the lesson
comes out stronger once it's fixed. Thanks for building this.
```

---

## Send it (3 steps)

1. In your fork folder: `git checkout -b fix-lesson22-stationary-distribution`
2. Copy `my-builds\contrib\pr-01-fix-stationary-distribution\stochastic.py` over `phases\01-math-foundations\22-stochastic-processes\code\stochastic.py`, then commit with the message above and push.
3. On GitHub, open a PR from that branch to `rohitg00/ai-engineering-from-scratch` and paste in the title and description.

Prefer no terminal? Edit the file straight on github.com in your fork (paste the fixed function), commit to a new branch there, and it will offer to open the PR for you.
