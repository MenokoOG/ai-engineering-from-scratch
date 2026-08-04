# Course Feedback — 05 Chain Rule & Autodiff

**Reviewed:** code/autodiff.py (run, exit 0; all built-in asserts and gradient checks passed), docs/en.md, quiz.json (no Julia files in this lesson's code folder)
**Verdict:** Clean

## Bugs & errors

None found. The Value engine's backward rules are all correct (verified against numerical gradients to ~1e-9 on five expressions, plus the script's own asserts). The XOR MLP trains from loss 4.15 to 0.18 in 100 steps and classifies all four inputs correctly. The docs' worked numbers check out (cos(4)*4 = -2.615 forward-mode example, tanh'(2) ≈ 0.0707, relu graph gradients 3 and 2).

## Nitpicks & suggestions

- `Value.backward()` builds the topo order with recursion. On deep graphs (long training chains, big expressions) this can hit Python's recursion limit. Worth a note or an iterative version.
- `__pow__` assumes a plain int/float exponent and `log()` has no domain guard for data <= 0; both fine for the lesson but worth flagging as extension exercises.
- There is no `__rtruediv__`, so `2 / x` fails while `x / 2` works. Minor asymmetry students may hit.
- The docs' XOR loop works only because `__radd__`/`__sub__` were defined in Step 4 (`sum(...)` starts at 0; targets are plain ints). A one-line pointer to that dependency would prevent copy-order errors.

## What's solid

- This is a faithful, correct micrograd-style engine, and the lesson verifies it three ways: manual math, numerical gradient checking, and (optionally) PyTorch. That's the right habit to teach.
- The forward-vs-reverse-mode explanation and all five quiz answers are accurate.
