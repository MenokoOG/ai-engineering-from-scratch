# 16 — Sampling Methods
> Generating is just choosing from a distribution — the art is in how you choose.

**Project:** One script that scales a toy language-model distribution with temperature, contrasts top-k and top-p filtering, demos the reparameterization trick numerically, runs Metropolis-Hastings MCMC on a bimodal target (including the too-big-proposal failure), and shows rejection sampling's acceptance rate collapsing with dimension.

## What I built
- Temperature scaling on an 8-word toy vocab: T=0.5 sharpens, T=2 flattens, T=100 is nearly uniform
- Top-k and top-p (nucleus) filters from scratch, showing top-p adapts to confidence and top-k does not
- Reparameterization trick: z = mu + sigma*eps gives a working Monte Carlo gradient (verified against the exact answer)
- Metropolis-Hastings sampler for a two-bump distribution, with acceptance-rate readouts for step sizes from 0.1 to 50
- Rejection sampling of a d-dimensional ball inside a cube, acceptance rate vs dimension table

## Main points learned
- Temperature divides the logits before softmax. Below 1 it exaggerates the leader; above 1 it evens things out.
- Temperature never changes the ranking of tokens, only how lopsided the probabilities are.
- Top-k keeps a fixed number of tokens; top-p keeps however many it takes to cover, say, 80% of the probability.
- You cannot backpropagate through a dice roll, but you can move the dice outside: z = mu + sigma*eps.
- Metropolis-Hastings needs the target only up to a constant, which is why it works when normalizing is impossible.
- MCMC step size is a dial: too small crawls and gets stuck in one bump; too large gets rejected and freezes.
- Rejection sampling dies in high dimensions because the target region becomes a vanishing sliver of the box.

## The algorithms, explained simply
**Temperature scaling.** Think of the probabilities as a leaderboard. Low temperature is like a winner-take-all bonus: the leader's advantage gets exaggerated. High temperature is like grading on a generous curve: everyone's chances even out. The order of the leaderboard never changes, only the gaps.

**Top-k vs top-p.** Top-k is a guest list with a fixed length: exactly k names, no matter what. Top-p is a guest list by importance: invite people until you've covered 80% of the "influence" in the room. When the model is confident that may be 1 token; when it's confused it may be 20. Top-p bends with the situation; top-k doesn't.

**Reparameterization trick.** You can't ask "how would this dice roll have changed if I'd tweaked the settings?" — a roll is a roll. The trick: roll a standard die first (eps), then compute the outcome as settings-times-roll (mu + sigma*eps). Now the randomness is locked in a separate box and the settings enter through plain arithmetic, which gradients can flow through.

**Metropolis-Hastings MCMC.** A hiker explores a foggy landscape where height means probability. From the current spot, propose a random step: always accept uphill, accept downhill sometimes (with probability equal to the height ratio). Tiny steps mean the hiker shuffles forever near one peak. Giant steps mean almost every proposal lands in a worthless valley and gets rejected, so the hiker just stands still.

**Rejection sampling.** Throw darts at a box that contains your target shape, keep only darts landing inside the shape. In 2D the circle fills 78% of the square — fine. In 15 dimensions the ball fills about a hundred-thousandth of a percent of the cube, so nearly every dart is wasted. Volume hides in the corners as dimensions grow.

## How this shows up in AI
- Every LLM decoding call uses this exact pipeline: logits / temperature, then top-k or top-p filter, then sample.
- Nucleus (top-p) sampling is the default in most chat APIs because it adapts to the model's confidence token by token.
- The reparameterization trick is the engine of VAEs and appears in diffusion models and stochastic policies.
- MCMC-style methods power Bayesian deep learning and some preference-sampling schemes; the step-size tradeoff is the classic tuning problem.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/16-sampling-methods
python3 project.py
```
