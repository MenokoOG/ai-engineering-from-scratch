# 04 — Calculus for ML

> The derivative tells you which way is downhill. Training is just walking downhill.

**Project:** Built numerical derivatives, multivariable gradients, and gradient descent from scratch. Showed why step size matters and verified analytic gradients with gradient checking.

## What I built
- Central difference derivative: `(f(x+h) - f(x-h)) / (2h)`
- Gradient of a 2-variable function, one dimension at a time
- Gradient descent on a simple loss with `w = w - lr * dL/dw`
- Step size demo: too small (slow), good (converges), too large (blows up)
- Gradient checking: analytic gradient vs numeric gradient on a tiny linear model

## Main points learned
- A derivative is just "how much does the output change if I nudge the input a tiny bit."
- A gradient is a list of derivatives, one per input. It points in the steepest uphill direction.
- To minimize a loss, step opposite the gradient. That is the whole idea of training.
- The central difference formula is more accurate than the one-sided version for the same h.
- Learning rate (step size) is the most important knob. Too big and the loss explodes.
- Gradient checking compares your hand-derived gradient to the numeric one. Tiny relative error means your math is right.
- Newton's method needs the Hessian (a matrix of all second derivatives). With millions of parameters that matrix is far too big, so deep learning sticks with first-order gradients.

## The algorithms, explained simply
**Central difference derivative.** Nudge the input a tiny bit up and a tiny bit down, and see how the output changes. It is like testing a seesaw by pressing gently on both sides instead of just one. Pressing both sides cancels out most of the measurement error.

**Gradient (multivariable).** Do the nudge test on each input separately while holding the others still. The results form an arrow that says "steepest way up is over there."

**Gradient descent.** Stand on a foggy hill and feel the slope under your feet. Take a small step downhill, then feel again. Repeat until the ground is flat. The learning rate is your stride length: baby steps take forever, giant leaps overshoot the valley.

**Gradient checking.** Before trusting your hand-derived formula, compare it to the slow-but-honest nudge test. If the two agree to many decimal places, your formula is correct. It is like checking mental math with a calculator.

## How this shows up in AI
- Every neural network is trained with the exact update rule built here: `w = w - lr * dL/dw`, just with millions of w's.
- Learning rate is the first hyperparameter anyone tunes when training an LLM; a bad one diverges exactly like the demo.
- Gradient checking is how autograd engines (like PyTorch's) are tested for correctness.
- Newton's method is skipped in deep learning because the Hessian is too big; optimizers like Adam approximate curvature cheaply instead.

## Run it
```
python3 project.py
```
