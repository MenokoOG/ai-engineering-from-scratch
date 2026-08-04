# Course Feedback — 04 Calculus for ML

**Reviewed:** code/derivatives.py (run, exit 0), docs/en.md, quiz.json (no Julia files in this lesson's code folder)
**Verdict:** Clean

## Bugs & errors

None found. Numerical derivatives match analytical to ~1e-9, the Hessian demos (saddle [[2,0],[0,-2]], bowl [[2,0],[0,2]], Rosenbrock [[802,-400],[-400,200]] at (1,1)) all match hand computation, the Taylor tables are correct, and the linear-regression gradients (dw = 2*error*x, db = 2*error) are right.

## Nitpicks & suggestions

- Docs "Derivatives by hand" table writes `f(x) = wx + b` then gives `f'(w) = x`. Mixing the variable of the function name (x) with the variable of differentiation (w) is sloppy notation; the surrounding text explains it, but the table alone reads oddly.
- Docs say "h = 0.0001 works well in practice" while every code sample uses h = 1e-7. Not wrong (both work for central differences), just inconsistent.
- The 200-epoch regression demo ends at w=2.08, b=0.73 vs the true (2, 1). Expected with lr=0.01, but a sentence noting the bias converges slowly would preempt "is this broken?" questions.

## What's solid

- The Hessian/eigenvalue treatment (saddle vs minimum, Newton vs gradient descent cost table) is accurate and unusually well connected to ML practice.
- All five quiz answers and explanations are correct, including the central-difference formula question.
