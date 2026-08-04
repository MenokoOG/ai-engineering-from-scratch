# 07 — Bayes' Theorem
> New evidence doesn't replace your beliefs — it updates them.

**Project:** One script with the classic rare-disease test calculator, a tiny from-scratch Naive Bayes spam classifier with Laplace smoothing, an MLE vs MAP coin demo, and step-by-step Bayesian updating from Beta(1,1) to Beta(8,4).

## What I built
- Rare-disease calculator: 1-in-10,000 disease, 99% accurate test, posterior under 1%
- The same result recomputed with plain counts over 1,000,000 people
- Naive Bayes spam filter on toy emails, shown breaking with zero counts and fixed by Laplace smoothing
- MLE vs MAP on a 3-heads-in-3-flips coin, plus proof they converge with lots of data
- Sequential Beta updating: flat prior plus 7 heads and 3 tails, printed one flip at a time

## Main points learned
- Bayes' rule combines what you believed before (the prior) with what the evidence says (the likelihood).
- A positive result from an accurate test can still mean you are probably fine, if the condition is rare.
- The trick: false positives from the huge healthy group outnumber true positives from the tiny sick group.
- Naive Bayes multiplies word probabilities, so a single never-seen word can zero out a whole class.
- Laplace smoothing fixes that by pretending every word was seen once more than it was.
- MLE trusts only the data; MAP adds a prior that acts like a few imaginary extra data points.
- With a Beta prior, updating is just counting: add heads to one number, tails to the other.

## The algorithms, explained simply
**Bayes' rule (disease test).** Think of screening a whole city. Almost everyone is healthy, so even a 1% error rate on the healthy crowd creates a big pile of false alarms. Your positive test most likely came from that pile, not from the tiny group of truly sick people. The rarer the disease, the more the false alarms dominate.

**Naive Bayes with Laplace smoothing.** The classifier asks: which folder (spam or not) makes these words least surprising? It naively treats each word as independent, like judging a fruit basket one fruit at a time. Smoothing gives every word a tiny head-start count, so one unfamiliar word cannot veto the whole verdict.

**MLE vs MAP.** MLE is a detective who only trusts the evidence on the table: 3 heads in 3 flips means the coin always lands heads. MAP is a detective with experience: coins are usually fair-ish, so 3 heads is probably a lucky streak. The prior works like imaginary earlier flips, and real data eventually outvotes it.

**Sequential Beta updating.** A Beta distribution is just a running scoreboard of heads vs tails, starting at 1-1 for "no idea." Each flip bumps one side of the score. Beta(8,4) simply means the scoreboard reads 7 heads and 3 tails on top of the starting point, and it doesn't matter what order they arrived in.

## How this shows up in AI
- Model evaluation is a disease test: a rare failure mode plus an imperfect detector means most flagged outputs are false alarms.
- Laplace smoothing is the ancestor of label smoothing and other tricks that stop models from assigning exact zeros.
- Weight decay and other regularizers are MAP in disguise: a prior saying "prefer small weights."
- RLHF and fine-tuning are sequential updating: start from prior beliefs (pretrained weights) and update with new evidence.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/07-bayes-theorem
python3 project.py
```
