# 12 — Tensor Operations
> A tensor is a labeled box of numbers; the labels do all the work.

**Project:** Built the numpy broadcasting rule from scratch and verified it against numpy, implemented the attention einsum 'bhtd,bhsd->bhts' with plain loops, walked through shapes/ranks and the NCHW image layout, and used strides to show exactly why transpose-then-view fails in PyTorch.

## What I built
- Shape/rank walkthrough from scalar up to a 4D attention tensor
- A from-scratch `broadcast_shape` function (right-align dims, 1s stretch), checked against `np.broadcast_shapes`, including (8,1,6,1) x (7,1,5) -> (8,7,6,5)
- The attention score einsum 'bhtd,bhsd->bhts' computed three ways: five nested loops, np.einsum, and batched matmul — all equal
- NCHW image batch demo: indexing one pixel, per-channel means, converting to NHWC
- Strides demo: transpose shares memory with swapped strides, so flattening it forces a copy — which is why torch's `.view()` refuses

## Main points learned
- Shape is the size along each axis; rank is just how many axes there are.
- Broadcasting rule: line shapes up from the right; each pair must match or one must be 1; the 1 stretches.
- (8,1,6,1) x (7,1,5) -> (8,7,6,5). The missing dim counts as 1, and every 1 stretches to match.
- In einsum, an index that appears in the inputs but not the output gets summed away. In 'bhtd,bhsd->bhts', that's d.
- PyTorch images are NCHW: batch, channels, height, width. TensorFlow historically used NHWC.
- Transposing never moves data — it just swaps strides (the step sizes used to walk memory).
- `.view()` promises zero-copy, but a transposed tensor's memory order can't be relabeled into a flat walk. Fix: `.contiguous().view()` or `.reshape()` (which copies when needed).

## The algorithms, explained simply
**Broadcasting.** Line up the two shapes from the right edge, like right-aligning two numbers before adding. Wherever one side has a 1 (or nothing), photocopy it along that axis until sizes match. It's how one bias vector gets added to a whole batch without writing a loop.

**Einsum.** Name every axis with a letter, then say which letters you want to keep. Repeated letters that vanish from the output get multiplied and summed — like a spreadsheet where you say "for every batch, head, query, and key, sum products over the feature column."

**Strides.** An array is one flat strip of memory plus a rulebook: "to move one step along this axis, jump this many bytes." Transpose just edits the rulebook. Like reading a grid column-first by changing your finger movement — the page never changes.

**View vs copy.** A view is a new label on the same memory — free but picky. A copy rewrites the numbers in a new order — always works, costs time and memory. `.view()` is the picky one; `.reshape()` quietly copies when it must.

## How this shows up in AI
- Attention is literally 'bhtd,bhsd->bhts': every query dotted with every key, per batch and head — the shape of the score matrix in every transformer.
- Broadcasting is how biases, masks, and layernorm stats apply across whole batches with no loops.
- Multi-head attention is a dance of reshape/transpose to split embeddings into heads — exactly where contiguity errors bite in real code.
- Conv layers, BatchNorm, and image pipelines all assume NCHW; mixing up the layout silently scrambles channels.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/12-tensor-operations
python3 project.py
```
