# 18 — Convex Optimization

> A convex problem has one valley; every downhill path leads home.

**Project:** Built a convexity checker (midpoint test), showed Newton's method solving a quadratic in one step vs 36 gradient descent steps, solved a constrained problem with projected gradient descent, and demoed KKT complementary slackness with a one-wall problem.

## What I built
- Convexity checker via the midpoint test: a chord (straight line between two points on the curve) must never dip below the curve
- Newton's method: exact minimum of 5x^2+3x+1 in 1 step, vs 36 steps of GD
- Projected gradient descent: minimize distance to (3,3) while staying inside the unit disk
- KKT complementary slackness demo: lambda = 0 when the constraint is not touched, lambda > 0 when the solution presses against it
- A plain-English summary of why SGD still works on non-convex deep learning losses

## Main points learned
- Convex means bowl-shaped: any local minimum is THE global minimum, so downhill always works.
- The midpoint test is a practical convexity check: pick two points, and the average of their heights must be at least the height at their average.
- Newton's method divides the slope by the curvature, so on a quadratic it jumps straight to the bottom in one step.
- Deep learning cannot use Newton because the curvature matrix (Hessian) for millions of weights is impossibly large.
- Constrained problems: take a normal gradient step, then snap back to the nearest allowed point. That is projected GD.
- Complementary slackness: each constraint either does not matter (lambda = 0) or is actively pushing on the solution (lambda > 0). Never both.
- Watch out: on a quadratic, GD with lr exactly 1/curvature IS a Newton step in disguise. Curvature and lr are deeply linked.

## The algorithms, explained simply
**Midpoint convexity test.** Stretch a rope between any two points on the curve. On a convex (bowl) function, the rope always hangs above the curve. If the rope ever ends up below, the function has a bump and is not convex.

**Newton's method.** GD asks "which way is downhill?" Newton also asks "how curved is the ground?" and uses that to compute the perfect stride. On a true bowl of known shape, it steps directly to the bottom. The price: measuring curvature is very expensive for big models.

**Projected gradient descent.** Walk downhill as usual, but you are fenced in. Any time a step lands outside the fence, get pulled back to the closest point on the fence. Eventually you settle at the fence spot nearest the true (forbidden) minimum.

**KKT complementary slackness.** Think of a ball rolling to the low point of a yard with a wall. If the low point is inside the yard, the wall exerts zero force (lambda = 0). If the low point is beyond the wall, the ball rests against the wall, which pushes back with real force (lambda > 0). "Force times gap = 0": there is only force when the gap is zero.

## How this shows up in AI
- Logistic regression, SVMs, and ridge/lasso are convex, which is why they train reliably and identically every run.
- Deep network losses are non-convex, yet SGD works: in high dimensions bad flat spots are mostly saddles that noise escapes, and big nets have many minima that are all good enough — we need a good low point, not the one perfect point.
- Projection is used in practice as gradient clipping and weight constraints: step, then snap back to the allowed set.
- Lagrange multipliers (the lambdas) show up in RLHF-style training as penalty weights, e.g. the KL penalty keeping a tuned model close to the base model.

## Run it
```
python3 project.py
```
