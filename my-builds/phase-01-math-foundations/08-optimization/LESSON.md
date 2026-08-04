# 08 — Optimization

> Same downhill idea, smarter footwork: momentum, per-weight step sizes, and a schedule.

**Project:** Built vanilla gradient descent, momentum, RMSProp, and Adam from scratch and raced them on an easy bowl and the hard Rosenbrock valley. Added a mini-batch SGD noise demo, a cosine annealing schedule, and a diverging learning-rate demo.

## What I built
- Four optimizers from scratch: vanilla GD, momentum, RMSProp, Adam
- A race with a comparison table (steps-to-converge, final loss) on two 2D functions: a bowl and Rosenbrock
- A too-large learning rate demo: lr just past the stability limit explodes
- Mini-batch SGD on linear regression with batch sizes 200/32/4, measuring loss "wiggle"
- Cosine annealing learning-rate schedule, beating a fixed small lr on the same budget

## Main points learned
- On an easy round bowl, plain GD is fine and even fastest. Fancy optimizers earn their keep on ugly landscapes.
- Rosenbrock's curved narrow valley wrecks plain GD; momentum and Adam cruise to the minimum.
- The stability rule of thumb: on a curvy direction, lr must be less than 2 divided by the curvature. Slightly over: divergence, not just slowness.
- Momentum remembers past gradients, so it powers through narrow valleys instead of zigzagging.
- RMSProp gives each weight its own step size based on how big its gradients have been.
- Adam = momentum + RMSProp + a bias fix for the first few steps. That is why it is the default.
- Mini-batch noise makes the loss curve wiggle, but the model still lands on the right answer. In deep nets that wiggle helps escape bad spots.

## The algorithms, explained simply
**Vanilla gradient descent.** Feel the slope, step downhill, repeat. Simple and honest, but in a narrow curved valley it bounces wall to wall and creeps along.

**Momentum.** A heavy ball rolling downhill. It keeps some of its previous speed, so side-to-side bounces cancel out and forward motion builds up. Great in valleys.

**RMSProp.** Every weight gets its own custom stride. Weights with consistently huge gradients get small careful steps; weights with tiny gradients get bigger ones. Like hiking with short steps on steep ground and long steps on flat ground.

**Adam.** The heavy ball AND the custom strides combined, plus a correction so the first few steps aren't timid. It is the "works out of the box" choice for training neural nets.

**Mini-batch SGD.** Instead of reading every review before updating your opinion, read a random handful each time. Cheaper per step, a bit jittery, same destination — and the jitter can shake you out of a rut.

**Cosine annealing.** Start with long strides to cover ground, then smoothly shorten your steps as you near home so you stop exactly at the door. The step size follows a smooth cosine curve from big to tiny.

## How this shows up in AI
- Adam (and its cousin AdamW) is the standard optimizer for training LLMs.
- LLM training uses warmup plus cosine decay — the exact schedule shape built here.
- "Loss went to NaN" during training is usually the Part 3 demo happening at scale: lr past the stability limit.
- Mini-batch noise is why training curves wiggle, and part of why SGD generalizes well in non-convex nets.

## Run it
```
python3 project.py
```
