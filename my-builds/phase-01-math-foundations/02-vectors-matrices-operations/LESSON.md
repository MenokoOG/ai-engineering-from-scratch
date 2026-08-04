# 02 — Vectors & Matrices Operations

> A neural network layer is just multiply, add, and clip — nothing more.

**Project:** Matrix multiply built with a plain triple loop, plus identity matrix, element-wise vs matrix product, a broadcasting rule checker, 2x2/3x3 determinants, and a full relu(W @ x + b) layer using only my own ops. Every result is asserted against numpy.

## What I built
- Triple-loop matmul with a shape check that rejects bad dimensions
- Identity matrix and proof that A @ I == A
- Side-by-side demo of element-wise product vs matrix product
- Broadcasting rule checker that agrees with numpy on every test case
- Determinant for 2x2 and 3x3, plus a det=0 example and why it matters
- One neural-net layer, relu(W @ x + b), computed with only my own code

## Main points learned
- To multiply (m x n) @ (n x p), the inner numbers must match. The result is (m x p).
- The identity matrix is the "times 1" of matrices: it changes nothing.
- Element-wise product pairs up matching cells. Matrix product mixes each row with each column.
- Broadcasting stretches a small array to fit a big one. Rule: line shapes up from the right; each dim must match or be 1.
- det=0 means the matrix squashes space flat and loses information. You cannot undo it (no inverse).
- One neural-net layer is just: mix the inputs (W @ x), shift them (+ b), keep the positives (relu).

## The algorithms, explained simply
**Triple-loop matmul.** For each cell of the answer, walk along a row of A and a column of B, multiplying pairs and adding them up. It is like computing a total bill: quantities from one list times prices from another.

**Broadcast rule checker.** Line the two shapes up at their right ends, like right-aligning two numbers. Compare column by column: equal is fine, a 1 stretches to fit, anything else is an error. It is like copying one row of a spreadsheet down every row so the sizes agree.

**Determinant.** A single number that says how much a matrix stretches area (2x2) or volume (3x3). Think of stamping a rubber square with the matrix: det is the size of the stamped shape. If det is 0, the square got flattened into a line — you can never recover it.

**The relu layer.** Multiply inputs by learned weights, add a learned offset, then set negatives to zero. Like a panel of judges: each judge (output) takes a weighted vote over the inputs, adds a personal bias, and only positive opinions get through.

## How this shows up in AI
- Matmul is the workhorse of deep learning; GPUs exist mostly to do it fast.
- Every linear layer in an LLM is exactly W @ x + b with a nonlinearity like relu after it.
- Broadcasting is how one bias vector is added to a whole batch of inputs at once.
- Shape errors are the most common deep-learning bug; the (m x n) @ (n x p) rule is the fix.

## Run it
```
python3 project.py
```
