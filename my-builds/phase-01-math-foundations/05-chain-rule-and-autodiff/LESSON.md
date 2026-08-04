# 05 — Chain Rule and Autodiff

> Backprop is just the chain rule, applied backwards through a graph, with good bookkeeping.

**Project:** Built a micrograd-style autodiff engine (the `Value` class) from scratch in Python, trained a tiny 2-layer network on XOR with it, and verified its gradients numerically. Also ported the core engine to dependency-free TypeScript.

## What I built
- `Value` class with `+`, `*`, `tanh`, `relu`, each op recording its own local backward rule
- `backward()` using a topological sort so gradients flow in the right order
- Gradient accumulation with `+=`, plus a demo of the bug you get with `=`
- A 2-layer neural net (4 hidden tanh units) trained on XOR using only this engine
- Gradient check: engine gradient vs central-difference numerical gradient
- `project.ts`: same core engine (add, mul, tanh, backward) in plain TypeScript

## Main points learned
- The chain rule says: to get the effect of a deep input on the final output, multiply the local slopes along the path.
- A computational graph is just a record of every operation and what fed into it.
- Each operation only needs to know its own local derivative. The engine chains them together.
- `backward()` must visit nodes in reverse topological order, so a node's gradient is complete before it passes gradient to its inputs.
- Gradients must accumulate with `+=`. If a value is used twice, it gets gradient from two paths, and `=` would erase one.
- Reverse mode (backprop) is used because one backward pass gives gradients for ALL parameters at once. That is perfect when there are millions of inputs and one loss.
- Always zero gradients before each training step, because `+=` would otherwise mix in last step's gradients.

## The algorithms, explained simply
**Computational graph.** Every math step gets written down like a receipt: what was computed and from what. Later you can walk the receipts backwards to see who influenced the final answer.

**Chain rule / backward pass.** Think of a factory line. If station B doubles the effect of station A, and station C triples the effect of B, then A affects the final product 6x. Backprop walks from the end of the line to the start, multiplying these local effects.

**Topological sort.** Before handing out blame, line everyone up so no one reports before all the people downstream of them have reported. Otherwise you would pass along an incomplete number.

**Gradient accumulation (+=).** If one worker's output is used in two places, their total influence is the sum of both. Using `=` is like only counting the last complaint about them and forgetting the first.

**Gradient checking.** Nudge one weight a tiny bit, watch the loss change, and compare against what the engine claims. If they match to many decimals, the engine is right.

## How this shows up in AI
- PyTorch's autograd is this exact design at scale: tensors instead of single numbers, same graph + backward + `+=` accumulation.
- This is why PyTorch has `optimizer.zero_grad()` — gradients accumulate by design, so you must clear them each step.
- Training an LLM is one giant backward pass through a computational graph with billions of nodes.
- Weight sharing (same weights reused at every token position) works only because gradients accumulate across all uses.

## Run it
```
python3 project.py
npx tsx project.ts
```
