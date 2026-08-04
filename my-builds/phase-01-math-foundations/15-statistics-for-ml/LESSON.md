# 15 — Statistics for ML
> "Significant" is not the same as "matters."

**Project:** One script that simulates why sample variance divides by n-1, computes a p-value from scratch with a permutation test, demonstrates the multiple-comparisons trap, shows how huge test sets make tiny differences "significant", and builds a bootstrap confidence interval for comparing two models.

## What I built
- Simulation showing divide-by-n underestimates variance and divide-by-(n-1) hits the true value
- Permutation test computing a p-value by shuffling group labels, plus a null control
- 20-tests-at-alpha-0.05 simulation matching the 1 - 0.95^20 = 64% false-positive theory, plus Bonferroni fix
- Two-proportion z-test showing a 0.03% accuracy gap flip from "not significant" to "significant" as n grows
- Paired bootstrap 95% confidence interval for the accuracy difference between two toy models

## Main points learned
- A p-value answers one question: how often would pure luck produce a result this extreme?
- Sample variance divides by n-1 because data sits suspiciously close to its own average, making spread look smaller than it is.
- Test 20 things at the 5% level and there is about a 64% chance something "wins" by luck alone.
- The fix for many tests is a stricter bar per test (Bonferroni: divide alpha by the number of tests).
- With a million test samples, even a meaningless 0.03% gap becomes statistically significant.
- Always ask two questions: is the effect real, AND is it big enough to care about?
- Bootstrap gives confidence intervals by resampling your own data — no formulas or normality assumptions needed.

## The algorithms, explained simply
**Bessel's correction (n-1).** Measuring spread around your sample's own average is like grading your own homework — everything looks a bit closer to correct than it really is. The sample mean was chosen to sit in the middle of your data, so the spread around it is slightly too small. Dividing by n-1 instead of n inflates the estimate just enough to fix the bias.

**Permutation test.** If the two groups really don't differ, then the group labels are meaningless stickers. So rip off all the stickers, shuffle them, and re-deal thousands of times. The p-value is just the fraction of shuffles that produced a gap as big as the real one. If shuffling almost never does it, the real gap probably isn't luck.

**Multiple comparisons.** Each test is a slot machine with a 1-in-20 jackpot of false alarm. Pull 20 slot machines and you'll probably hit at least one jackpot somewhere. That "winning" model config is often just the lucky machine.

**Bootstrap confidence interval.** You only have one test set, but you can fake having thousands: redraw examples from it with replacement, like pulling names from a hat and putting each name back. Recompute the model gap on each redraw. The range covering the middle 95% of those gaps tells you how much the gap wobbles. If zero is inside the range, the models might be tied.

## How this shows up in AI
- Benchmark leaderboards with tiny gaps between models are exactly the tiny-effect trap: check effect size, not just rank.
- Hyperparameter sweeps are multiple comparisons: the best of 50 runs is partly a lucky run, so re-validate the winner on fresh data.
- Paired bootstrap on per-example scores is the standard honest way to say "model B beats model A" on an eval set.
- Reporting variance across seeds needs the n-1 correction, since eval runs are small samples.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/15-statistics-for-ml
python3 project.py
```
