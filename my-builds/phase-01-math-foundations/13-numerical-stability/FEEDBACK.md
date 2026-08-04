# Course Feedback — 13-numerical-stability

**Reviewed:** code/numerical.py (ran, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Bugs found

## Bugs & errors

1. **code/numerical.py, `demo_nan_inf` (~line 477).** The output line `max = nan (comparison with nan is always False)` is hardcoded text, and it is wrong for the code shown. In pure Python, `max([1.0, 2.0, float('nan'), 4.0, 5.0])` returns `5.0` (verified). The nan-poisons-max behavior is a numpy thing, not a Python `max()` thing. Fix: actually compute and print `max(values)`, and explain that the result depends on where the nan sits (nan wins only if it is first). Or say "in numpy, np.max would return nan."

2. **code/numerical.py, `demo_common_bugs` Bug 3 (~line 636).** The demo claims "Variance underflow with large-mean data" but its own output shows the naive formula getting the exact answer: `Naive: 2.000000 (error: 0.00e+00)`. The chosen data (integers around 1e8, float64) is too benign — the demo fails to demonstrate the bug it names. Fix: use a larger mean (e.g. 1e9 with fractional values) or run in float32 so the naive method visibly fails. Note Demo 2 (mean 1e6) does show a small error, so only Bug 3 is broken.

3. **docs/en.md, softmax section (~lines 183-188).** Garbled leftover editing text: "These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually: ...". This reads like the author arguing with themselves, and the inequality as written is false (2.69e43 is greater than 3.4e38, so it does overflow float32). Fix: replace with a plain statement: exp(100) ≈ 2.7e43 exceeds the float32 max (~3.4e38), so it overflows; the float32 limit is exp(x) for x ≈ 88.7.

4. **docs/en.md, Build It Steps 2 and 3.** Two code comments are false for the Python code shown. Step 2: `# softmax_naive(dangerous_logits) would return [nan, nan, nan]` for logits [100, 101, 102] — in Python (float64) the naive softmax works fine; the lesson's own demo output prints matching naive results. Step 3: `# logsumexp_naive(large) returns inf` for [500, 501, 502] — false; math.exp(500) ≈ 1.4e217 fits in float64 and the naive version returns 502.4076 (verified). Both claims are true only in float32. Fix: say "would fail in float32; in Python's float64 the danger starts near logits of 710."

5. **docs/en.md, Exercise 2.** "Find the smallest positive float32 value x such that `1.0 + x == 1.0`" is backwards. Every tiny positive x satisfies `1.0 + x == 1.0`; the smallest such x is meaningless. Machine epsilon is (informally) the smallest x with `1.0 + x != 1.0`, which is how the lesson's own Key Terms table defines it. Fix the exercise wording to match.

6. **code/numerical.py, Demo 14 table (~line 683).** `float8 ... Max Value 240 ... (H100+)`. The OCP FP8 E4M3 format used by H100 has max value 448 (240 is the max of the E4M3FNUZ variant used elsewhere). With the "H100+" framing, 448 is the right number.

## Nitpicks & suggestions

- Demo 2's "subtracting nearly equal numbers" prints "Relative error: 0.0%" in float64, while the docs promise a 19.2% error (a float32 figure). The demo undercuts its own point; either simulate float32 or explain why float64 hides it here.
- `demo_nan_inf` prints `1.0 / 0.0 = inf` and `0.0 / 0.0 = nan` as if computed, but Python raises ZeroDivisionError for both; the printed values are hardcoded `float('inf')`/`float('nan')`. Worth a caveat (IEEE semantics vs Python semantics).
- Demo 13 Bug 1 describes a log(0) crash but never hits it: probs[1] = 1.38e-87 is still representable in float64, so the safe branch prints -200. Use logits like [1000, -1000, -1000] to actually trigger the crash branch.
- Demo 6 prints `Loss: -0.0000000000` (negative zero) for the confident-correct case; cosmetic but confusing for a loss.
- The 0.1 + 0.2 section quotes float32 stored values (0.100000001490116) then shows the float64 Python result (0.30000000000000004); the two representations are silently mixed.

## What's solid

- All the stable implementations (softmax, log-sum-exp, log-softmax, cross-entropy, sigmoid, BCE-with-logits, Kahan, Welford) are correct — I checked the math and the gradient-check demo passes/fails exactly as designed.
- Quiz answers and explanations are all correct, and the loss-scaling / bfloat16-vs-float16 material matches real practice.
