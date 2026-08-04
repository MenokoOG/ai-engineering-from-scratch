"""Lesson 21: Graph theory from scratch.
Graph class, BFS/DFS, Laplacian + connected components, GNN message passing,
spectral clustering with the Fiedler vector.
"""
from collections import deque

import numpy as np

rng = np.random.default_rng(21)


class Graph:
    def __init__(self, num_nodes):
        self.n = num_nodes
        self.adj = {i: [] for i in range(num_nodes)}

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def adjacency_matrix(self):
        a = [[0] * self.n for _ in range(self.n)]
        for u, nbrs in self.adj.items():
            for v in nbrs:
                a[u][v] = 1
        return a

    def bfs_shortest_path(self, start, goal):
        """Queue-based BFS: fewest-edges path in an unweighted graph."""
        queue = deque([start])
        parent = {start: None}
        while queue:
            node = queue.popleft()
            if node == goal:
                path = []
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]
            for nbr in self.adj[node]:
                if nbr not in parent:
                    parent[nbr] = node
                    queue.append(nbr)
        return None

    def dfs_order(self, start):
        """Explicit-stack DFS: visits one branch as deep as possible first."""
        stack = [start]
        seen = set()
        order = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            for nbr in reversed(self.adj[node]):
                if nbr not in seen:
                    stack.append(nbr)
        return order

    def degree_matrix(self):
        d = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            d[u][u] = len(self.adj[u])
        return d

    def laplacian(self):
        a = self.adjacency_matrix()
        d = self.degree_matrix()
        return [[d[i][j] - a[i][j] for j in range(self.n)] for i in range(self.n)]


print("=" * 60)
print("DEMO 1: Graph class, adjacency list + adjacency matrix")
print("=" * 60)
edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
g = Graph(6)
for u, v in edges:
    g.add_edge(u, v)
a = np.array(g.adjacency_matrix())
expected = np.zeros((6, 6), dtype=int)
for u, v in edges:
    expected[u, v] = expected[v, u] = 1
print("Edges:", edges)
print("Adjacency list:", dict(g.adj))
print("Adjacency matrix:")
print(a)
assert np.array_equal(a, expected) and np.array_equal(a, a.T)
print("PASS: matrix matches edge list and is symmetric (undirected).")

print()
print("=" * 60)
print("DEMO 2: BFS (queue -> shortest path) vs DFS (stack -> deep dive)")
print("=" * 60)
path = g.bfs_shortest_path(0, 5)
print(f"BFS shortest path 0 -> 5: {path} ({len(path) - 1} edges)")
assert len(path) - 1 == 3
assert all(a[path[i], path[i + 1]] == 1 for i in range(len(path) - 1))
order = g.dfs_order(0)
print(f"DFS visit order from 0:  {order}")
assert sorted(order) == list(range(6))
assert order[:4] == [0, 1, 3, 5], "DFS should dive down branch 0-1-3-5 first"
print("PASS: BFS found a 3-edge path; DFS went deep before backtracking.")

print()
print("=" * 60)
print("DEMO 3: Laplacian L = D - A, zero eigenvalues = # components")
print("=" * 60)
g2 = Graph(5)
for u, v in [(0, 1), (1, 2), (0, 2), (3, 4)]:
    g2.add_edge(u, v)
lap = np.array(g2.laplacian())
eigvals = np.linalg.eigvalsh(lap)
num_zero = int(np.sum(np.abs(eigvals) < 1e-8))
print("Graph: triangle {0,1,2} plus separate edge {3,4} -> 2 components")
print("Laplacian eigenvalues:", np.round(eigvals, 4))
print(f"Zero eigenvalues: {num_zero}")
assert num_zero == 2
lap1 = np.array(g.laplacian())
assert int(np.sum(np.abs(np.linalg.eigvalsh(lap1)) < 1e-8)) == 1
print("PASS: 2 zero eigenvalues for 2 components (and 1 for the connected graph).")

print()
print("=" * 60)
print("DEMO 4: One round of GNN-style message passing")
print("=" * 60)
h = rng.standard_normal((6, 2))
w = rng.standard_normal((2, 2))


def relu(x):
    return np.maximum(x, 0.0)


h_new = np.zeros_like(h)
for v in range(g.n):
    mean_nbr = np.mean([h[u] for u in g.adj[v]], axis=0)
    h_new[v] = relu(w @ mean_nbr)

deg_inv = np.diag(1.0 / a.sum(axis=1))
h_new_vec = relu((deg_inv @ a @ h) @ w.T)
print("h_new[v] = relu(W @ mean(neighbor features)) computed with loops:")
print(np.round(h_new, 4))
assert np.allclose(h_new, h_new_vec, atol=1e-10)
print("PASS: loop version matches vectorized relu(D^-1 A H W^T).")
print("Each node's new feature now mixes in info from its 1-hop neighborhood.")

print()
print("=" * 60)
print("DEMO 5: Spectral clustering via the Fiedler vector")
print("=" * 60)
g3 = Graph(6)
cluster_edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
for u, v in cluster_edges:
    g3.add_edge(u, v)
lap3 = np.array(g3.laplacian(), dtype=float)
vals, vecs = np.linalg.eigh(lap3)
fiedler = vecs[:, 1]
labels = (fiedler > 0).astype(int)
print("Graph: two triangles {0,1,2} and {3,4,5} joined by one bridge edge 2-3")
print("Fiedler vector (eigenvector of 2nd-smallest eigenvalue):", np.round(fiedler, 3))
print("Sign-based cluster labels:", labels.tolist())
assert labels[0] == labels[1] == labels[2]
assert labels[3] == labels[4] == labels[5]
assert labels[0] != labels[3]
print("PASS: the sign of the Fiedler vector splits the two obvious clusters.")

print()
print("ALL GRAPH DEMOS PASSED")
