# Course Feedback — 21-graph-theory

**Reviewed:** code/graph_theory.py (ran with python3, exit 0), docs/en.md, quiz.json (no Julia files in this lesson)
**Verdict:** Minor issues

## Bugs & errors

1. **docs/en.md, "Ship It" section vs code/graph_theory.py.** The docs say the lesson produces `outputs/skill-graph-analysis.md`, but graph_theory.py contains no file-writing code at all (verified: no `open`/`write` anywhere). Running the code as instructed never creates the promised output. Fix: either add a `write_skill_output()` function like lessons 19/20 have, or drop the Ship It claim.

## Nitpicks & suggestions

1. docs/en.md, Step 1: the doc's `degree_matrix` uses `self.degree(i)` (edge count) while the code file uses `weighted_degree(i)` (sum of weights). Identical for unweighted graphs, but the doc version gives a wrong Laplacian for weighted graphs. The code file's version is the right one.
2. docs/en.md, Key Terms: "Fiedler value: the smallest non-zero eigenvalue of L." The standard definition is the second-smallest eigenvalue. These coincide for connected graphs but differ for disconnected ones (where the second-smallest is 0).
3. docs/en.md, graph-type table: "Directed, weighted — Web page links (PageRank scores)" is an odd example; the web graph is directed but typically unweighted, and PageRank scores are node outputs, not edge weights.

## What's solid

- Everything the code demonstrates is correct and verified in output: BFS distances, the zero-eigenvalue count matching 3 connected components, spectral clustering splitting the two cliques perfectly (Fiedler value 0.2984), and PageRank correctly ranking the bridge nodes (2, 7) highest.
- The GCN connection (A_hat = A + I, symmetric normalization, relation to L_sym) is stated accurately — a nice bridge from the math to real GNNs.
