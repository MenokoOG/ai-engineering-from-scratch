# Course Feedback — 12 Tensor Operations

**Reviewed:** code/tensors.py (ran with python3), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Build It" Step 2.** Claims: "a bias vector `(D,)` added to a batch `(B, T, D)` needs unsqueezing to `(1, 1, D)`." False — and it contradicts the doc's own broadcasting section two headings earlier ("Fewer dimensions get padded with 1s on the left"). A `(D,)` vector broadcasts against `(B, T, D)` directly with no unsqueeze; NumPy and PyTorch both do this, and the course's own `demo_broadcasting_numpy` adds a `(3,)` bias to `(4, 3)` with no reshaping. Unsqueeze is needed when the axis you want is NOT the last one (e.g., adding a per-token `(T,)` vector along axis 1 requires `(T, 1)`). Fix: change the example to a non-trailing axis, or say the unsqueeze is optional/for clarity.

## Nitpicks & suggestions

1. The custom `Tensor` class defines `__add__`/`__mul__`/`__sub__` but no reflected ops (`__radd__`, `__rmul__`), so `t * 2` works but `2 * t` raises TypeError. Worth one line each, since students will try it.
2. `squeeze(dim)` on a non-size-1 dim silently returns an unchanged copy. PyTorch does the same shape-wise, so this is fine, but a docstring note would help; some might expect an error.
3. `__getitem__` only supports full indexing (all axes at once). The error message says so, which is good; the docs never mention the limitation.
4. "Ship It" references `outputs/prompt-tensor-shapes.md` and `outputs/prompt-tensor-debugger.md`; neither exists in the lesson folder.
5. Docs Step 7 attention snippet is an excerpt (K, V, W_o, softmax appear undefined). It does say the full version lives in `demo_attention_einsum()`, but a "(excerpt)" marker would prevent copy-paste confusion.

## What's solid

- The from-scratch Tensor class is genuinely correct: strides, reshape with -1 inference, permute (verified the (2,0,1) permutation output against PyTorch semantics), flatten with start_dim, and axis-wise sum all behave like the real frameworks.
- The einsum gallery, multi-head attention shape walkthrough (all shapes verified in the run), memory-layout/strides demos, and the quiz (all 5 answers correct) tie tensor mechanics directly to transformer code. The F-order stride example in the docs, (1, 2) for a 2x3 matrix, matches the actual NumPy output.
