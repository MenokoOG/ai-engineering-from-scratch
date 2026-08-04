# PR #1 (new target) — lesson 17 wrong least-squares answer

The lesson 22 fix got beaten to the punch — the maintainer already patched it upstream. This one is still live on upstream main (verified 2026-08-04). One character, verified three ways.

---

## The fix

File: `phases/01-math-foundations/17-linear-systems/docs/en.md`

Find this line (only one occurrence, in the normal-equations worked example):

```
Solve: x = [1.5, 1.7]
```

Change it to:

```
Solve: x = [1.5, 1.6]
```

## PR title

```
Fix: wrong least-squares answer in Phase 1, Lesson 17 docs
```

## Commit message

```
fix least-squares worked example answer in phase 1 lesson 17
```

## PR description

```
Working through Phase 1, Lesson 17, I ran the normal-equations worked
example myself and the doc's answer doesn't check out.

The setup is right:

  A^T A = [[4, 10], [10, 30]],  A^T b = [22, 63]

But solving that system gives x = [1.5, 1.6], not the [1.5, 1.7]
printed in the doc. Residuals back it up: ||Ax - b|| is 0.447 with
[1.5, 1.6] and 0.707 with [1.5, 1.7]. numpy's lstsq lands on
[1.5, 1.6] too.

One character in docs/en.md. The doc points out this example IS
linear regression, so the slope being off by 0.1 is worth catching
before it sticks in somebody's head.

Running the math by hand on every lesson as I go -- solid course.
```

---

## Send it from GitHub (you're already there)

1. On your fork's main page, hit **Sync fork** first so you're editing current code.
2. Open `phases/01-math-foundations/17-linear-systems/docs/en.md`, click the pencil, change the 7 to a 6 on the `Solve: x = [1.5, 1.7]` line.
3. Commit changes → **Create a new branch** → name it `fix-lesson17-least-squares-answer` → Propose changes → paste the title and description above into the PR.

Docs-only change: no parser concerns, no site/build.js needed.
