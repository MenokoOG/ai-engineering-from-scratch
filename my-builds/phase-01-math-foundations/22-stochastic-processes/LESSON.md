# 22 — Stochastic Processes

> Randomness with rules: memoryless hops, sqrt(n) drift, and noise you can run backwards.

**Project:** Built a Markov weather simulator, a random-walk ensemble that recovers the sqrt(n) law, the stationary distribution computed two independent ways, Langevin sampling of a double-well energy, and a toy diffusion-model forward process.

## What I built
- Markov chain weather sim (Sunny/Cloudy/Rainy). Verified tomorrow depends only on today, not yesterday.
- 4000 random walks of 10,000 steps. Fitted the growth exponent: distance ~ n^0.498.
- Stationary distribution via power iteration on the matrix vs long-run visit counts. They match.
- Langevin dynamics on U(x) = (x^2 - 1)^2. Low temperature freezes into one valley; higher temperature hops between both.
- Diffusion forward process: 2D blobs get noised step by step into a pure gaussian, with SNR tracked down.

## Main points learned
- Markov property: the future depends only on the present state. History adds nothing.
- A random walk drifts only as sqrt(n): 100x more steps buys just 10x more distance.
- The stationary distribution is where a chain settles; it satisfies pi @ P = pi.
- Two totally different methods (matrix math vs counting visits) give the same stationary answer.
- Langevin dynamics = roll downhill + random kicks. Temperature controls how adventurous the kicks are.
- T -> 0 turns Langevin into plain gradient descent: it just finds the nearest minimum.
- Diffusion models destroy data with scheduled noise; the model learns to undo one small step at a time.

## The algorithms, explained simply
**Markov chain.** A board game where your next move depends only on the square you are on, never on how you got there. Each square has fixed dice odds for where you go next. Simulate by rolling those dice over and over.

**Random walk sqrt(n) law.** Flip a coin, step left or right, repeat. Steps mostly cancel, so you wander outward slowly, like a drunk person who ends up only sqrt(n) blocks from the bar after n steps. We measured it and fit the exponent: 0.498.

**Stationary distribution.** Run the weather for years and count the sunny days: that long-run share is the stationary distribution. Equivalently, keep multiplying a guess by the transition matrix until it stops changing. Both roads reach the same place, which is the point.

**Langevin dynamics.** A ball rolling downhill in an energy landscape while being randomly jiggled. Cold jiggling: the ball settles in the nearest valley and stays. Warm jiggling: the ball occasionally kicks over the hill and visits both valleys, sampling the whole landscape.

**Diffusion forward process.** Take a clean photo and add a little static, then a little more, on a fixed schedule until only static remains. Signal-to-noise falls at every step, by design. The trained model is the reverse: it looks at static and predicts the small amount of noise to remove, step by step back to a clean sample.

## How this shows up in AI
- Diffusion models (Stable Diffusion, etc.) are literally this forward noising plus a learned reverse denoiser.
- Langevin dynamics is how score-based models and some samplers draw from a learned energy landscape.
- MCMC (Markov chain Monte Carlo) uses Markov chains whose stationary distribution is the target, for Bayesian inference.
- Random-walk sqrt(n) thinking explains noise accumulation in SGD and exploration in RL.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/22-stochastic-processes
python3 project.py
```
