# 06 — Probability and Distributions
> Probability is how models say "I'm not sure" with numbers.

**Project:** A single script that demos PMF vs PDF with a die and a gaussian, simulates the Central Limit Theorem, breaks and then fixes softmax with the max-subtraction trick, shows cross-entropy loss as -log(p_true), and proves why log-probs beat raw probabilities.

## What I built
- Fair-die PMF (sums to 1) vs narrow gaussian PDF (density above 1, area equals 1)
- Central Limit Theorem simulation: means of uniform draws form a bell curve (text histogram)
- Naive softmax that overflows on big logits, and a stable version that subtracts the max
- Cross-entropy loss table showing -log(p_true) for confident-right vs confident-wrong
- Underflow demo: multiplying 1000 probabilities gives 0.0, summing their logs works fine

## Main points learned
- A PMF gives real probabilities to separate outcomes, like die faces. It sums to 1.
- A PDF gives density, not probability. Density can be above 1. Only areas (ranges) are probabilities.
- The chance of a continuous value landing on one exact number is zero.
- Averages of almost anything become bell-shaped as you average more values. That is the Central Limit Theorem.
- Softmax must subtract the max logit first, or exp() overflows on big numbers. The answer does not change.
- Cross-entropy loss only looks at the probability the model gave the correct answer. Being confidently wrong costs a lot.
- Multiplying many probabilities underflows to zero. Adding their logs keeps all the information.

## The algorithms, explained simply
**PMF vs PDF.** A PMF is like a vending machine price list: each item (die face) has its own price (probability). A PDF is like population density on a map: the number at one point is not a count of people, but adding it up over an area gives you the count. Skinny, tall bells can have density above 1 and still be valid.

**Central Limit Theorem.** Take any messy source of randomness, average a bunch of draws, and repeat. The averages pile up in a bell shape, like many small coin-flip nudges canceling out. The bell also gets narrower as you average more draws.

**Stable softmax.** Softmax turns scores into probabilities by exponentiating and dividing. exp() of a big score blows past what a computer can store, like a calculator showing ERROR. Subtracting the biggest score first shifts everything down without changing the final ratios, like measuring heights from the tallest person instead of from sea level.

**Cross-entropy loss.** The loss is just "how surprised was the model by the right answer." If it gave the truth 99% it barely gets punished. If it gave the truth 1% it gets punished hard, because -log of a tiny number is huge. It is a surprise meter.

**Log-probabilities.** A sentence's probability is many small numbers multiplied together, which shrinks to nothing a computer can represent. Logs turn multiplication into addition, like using orders of magnitude instead of writing out all the zeros. The comparisons stay exactly the same.

## How this shows up in AI
- Every LLM ends in a softmax over the vocabulary, and every real implementation uses the max-subtraction trick.
- LLM pretraining loss is cross-entropy on the next token: predict the true token with high probability or pay.
- Perplexity and sequence scores are computed as sums of log-probs, never raw products.
- The CLT is why averaging over big batches gives stable, roughly gaussian loss and gradient estimates.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/06-probability-and-distributions
python3 project.py
```
