# 21 — Graph Theory

> Graphs are dots and lines; a few matrices turn them into things you can compute with.

**Project:** Built a Graph class (adjacency list + matrix), BFS and DFS, the graph Laplacian with a connected-components check, one round of GNN message passing, and spectral clustering with the Fiedler vector. A TypeScript version covers the Graph class, BFS, and the adjacency matrix.

## What I built
- Graph class storing edges two ways: adjacency list (fast neighbor lookup) and adjacency matrix (math-ready).
- BFS with a queue: finds the shortest path 0 -> 5 in an unweighted graph.
- DFS with a stack: dives deep down one branch before backtracking.
- Degree matrix and Laplacian L = D - A. Counted zero eigenvalues = number of connected pieces.
- One GNN message-passing round: each node averages its neighbors' features, applies weights and ReLU.
- Spectral clustering: the Fiedler vector's signs split two triangles joined by a bridge, perfectly.
- `project.ts`: the same tiny graph in dependency-free TypeScript.

## Main points learned
- Adjacency matrix entry A[i][j] = 1 just means "there is an edge from i to j."
- BFS uses a queue and explores in rings, so the first time it reaches a node is the shortest path.
- DFS uses a stack and commits to one branch; good for exploring, not for shortest paths.
- The Laplacian L = D - A (degrees minus connections) encodes the graph's shape as numbers.
- Count L's zero eigenvalues and you get the number of disconnected islands. Verified: 2 islands, 2 zeros.
- GNN message passing = each node updates itself using a summary of its neighbors.
- The Fiedler vector (2nd-smallest Laplacian eigenvector) finds the graph's natural weak spot to cut.

## The algorithms, explained simply
**BFS.** Like ripples from a stone: visit everyone 1 step away, then 2 steps, then 3. A queue (first in, first out) keeps the rings in order. Because rings expand evenly, the first route found is the fewest-edges route.

**DFS.** Like exploring a maze by always taking the next corridor until you hit a dead end, then backing up. A stack (last in, first out) remembers where to back up to. It visits everything but takes scenic routes.

**Laplacian components check.** The Laplacian is a bookkeeping matrix: how many friends each node has, minus who they are. Each fully separate island contributes exactly one zero eigenvalue, like each isolated room having its own silence. So counting zeros counts islands.

**GNN message passing.** Each person updates their opinion by averaging their friends' opinions, then passing it through a shared filter (weights + ReLU). One round mixes 1-hop info; stacking rounds spreads info further, like gossip.

**Spectral clustering.** Ask the Laplacian for its gentlest non-flat vibration mode (the Fiedler vector). The graph splits where that vibration changes sign, which lands on the weakest bridge between communities. Positive signs form one cluster, negative the other.

## How this shows up in AI
- GNNs (graph neural networks) power recommendations, molecule property prediction, and fraud detection using exactly this message-passing step.
- Spectral clustering and Laplacian eigenvectors underlie community detection and graph positional encodings for graph transformers.
- BFS/DFS are the backbone of knowledge-graph traversal and agent planning searches.
- Attention in Transformers is message passing on a fully connected graph of tokens.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/21-graph-theory
python3 project.py
npx tsx project.ts
```
