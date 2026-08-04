# Course Feedback — 09 Information Theory

**Reviewed:** code/information_theory.py (ran with python3), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **code/information_theory.py, "PERPLEXITY IN LANGUAGE MODELS" demo (lines ~316-333).** The demo's "model" is random Gaussian logits, which gives avg CE = 4.397 nats and perplexity 81.2. The uniform-random baseline over the 50-token vocab is ln(50) = 3.91 nats, i.e., perplexity 50. So the demo model is worse than random, yet the printout says "The model is better than random if perplexity < vocab size" without noting that this model fails that test. A student reading the output will see 81 > 50 and be confused. Verified by running. Fix: either bias the logits toward the true tokens so the model beats the baseline, or add a line stating this random model is worse than uniform (random logits waste probability on confident wrong guesses).

## Nitpicks & suggestions

1. Docs, label smoothing: the formula `L = (1-eps) * CE(hard, pred) + eps * H_uniform(pred)` is correct only if "H_uniform(prediction)" means cross-entropy between the uniform distribution and the prediction. The notation reads like an entropy. Recomputed: the decomposition holds with CE(uniform, pred). Clearer to write it that way.
2. Docs Step 1 snippet: `information_content` returns 0.0 for p > 1 instead of raising. Silently accepting invalid probabilities hides bugs. Same in the code file (`p >= 1` returns 0.0).
3. The feature-selection MI bars use `"#" * int(mi * 200)`, printing a 117-character bar for the strong feature. Cosmetic; wraps on narrow terminals.
4. Use It: `np_cross_entropy` will emit a divide-by-zero warning / return inf if q has a zero where p > 0. The scratch version handles this; the NumPy version doesn't. Worth a comment.

## What's solid

- Every formula and number I checked is right: entropy values, CE = H + KL verified in output, KL asymmetry, conditional/joint entropy identities (H(X,Y) = H(X) + H(Y|X) confirmed numerically), and CE = NLL shown to be exactly identical.
- The label smoothing and MI feature-selection demos connect the theory to real ML practice well, and all 5 quiz answers and explanations are correct.
