# 09 — Information Theory
> Surprise is measurable, and loss functions are just measured surprise.

**Project:** Built entropy, cross-entropy, KL divergence, perplexity, and mutual information from scratch in numpy. Verified the key identity CE = H(p) + KL(p||q) numerically and showed why correlation misses what mutual information catches.

## What I built
- Entropy calculator, tested on fair vs loaded vs certain dice
- Cross-entropy and KL divergence, with the identity CE = H(p) + KL(p||q) checked numerically
- A table proving CE and KL always differ by the constant H(p), so the same model minimizes both
- Perplexity = exp(cross-entropy) demo on a toy 50-word language model
- Histogram-based mutual information, compared to Pearson correlation on y = x^2

## Main points learned
- Entropy measures average surprise. Uniform (fair die) = max surprise. Certainty = zero surprise.
- Cross-entropy is the surprise you feel when reality follows p but you bet using model q.
- KL divergence is the extra surprise caused by using the wrong model. It is never negative, and zero only when the models match.
- CE = H(p) + KL. Since H(p) is fixed by the data, training on cross-entropy is secretly training on KL.
- Perplexity is exp(cross-entropy). Perplexity 50 means the model is as confused as picking among 50 equally likely words.
- Pearson correlation only sees straight-line relationships. y = x^2 gets correlation near 0.
- Mutual information sees any dependence, linear or not. It is near 0 only for truly independent variables.

## The algorithms, explained simply
**Entropy.** Add up how surprising each outcome is, weighted by how often it happens. Like rating a news channel: a channel that always says "sunny" carries no information; one where anything can happen carries a lot.

**Cross-entropy.** Measure surprise using your model's odds while reality rolls its own dice. Like a gambler using the wrong betting sheet: the worse the sheet matches reality, the more they lose per bet.

**KL divergence.** The gap between your surprise with the wrong model and the surprise you'd have with the true one. It is the "penalty for believing the wrong thing" — cross-entropy minus the unavoidable part.

**Perplexity.** Turn average surprise back into a head-count of equally likely choices. "Perplexity 50" reads as: the model feels like it's guessing among 50 words every step.

**Mutual information.** Compare the real joint behavior of two variables to what you'd see if they ignored each other. Any gap, in any shape, counts — like noticing two coworkers always take lunch at related times, even if the pattern is weird.

## How this shows up in AI
- Every LLM is trained by minimizing cross-entropy on next-token prediction, which is the same as minimizing KL to the true text distribution.
- Perplexity is the standard scoreboard for language models; lower means less confused.
- KL divergence is the regularizer in RLHF (keep the tuned model close to the base model) and in VAEs.
- Mutual information guides feature selection and shows why nonlinear signals survive that correlation-based filters would throw away.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/09-information-theory
python3 project.py
```
