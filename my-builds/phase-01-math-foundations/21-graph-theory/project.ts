// Lesson 21 (TypeScript): dependency-free Graph class with BFS shortest path
// and adjacency matrix, on the same tiny graph as project.py.
// Run: npx tsx project.ts

class Graph {
  n: number;
  adj: Map<number, number[]>;

  constructor(numNodes: number) {
    this.n = numNodes;
    this.adj = new Map();
    for (let i = 0; i < numNodes; i++) this.adj.set(i, []);
  }

  addEdge(u: number, v: number): void {
    this.adj.get(u)!.push(v);
    this.adj.get(v)!.push(u);
  }

  adjacencyMatrix(): number[][] {
    const a: number[][] = Array.from({ length: this.n }, () =>
      new Array(this.n).fill(0),
    );
    for (const [u, nbrs] of this.adj) {
      for (const v of nbrs) a[u][v] = 1;
    }
    return a;
  }

  bfsShortestPath(start: number, goal: number): number[] | null {
    const queue: number[] = [start];
    const parent = new Map<number, number | null>([[start, null]]);
    let head = 0;
    while (head < queue.length) {
      const node = queue[head++]; // pop from the front = queue behavior
      if (node === goal) {
        const path: number[] = [];
        let cur: number | null = node;
        while (cur !== null) {
          path.push(cur);
          cur = parent.get(cur)!;
        }
        return path.reverse();
      }
      for (const nbr of this.adj.get(node)!) {
        if (!parent.has(nbr)) {
          parent.set(nbr, node);
          queue.push(nbr);
        }
      }
    }
    return null;
  }
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
}

const edges: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 5],
];
const g = new Graph(6);
for (const [u, v] of edges) g.addEdge(u, v);

console.log("Edges:", JSON.stringify(edges));
const a = g.adjacencyMatrix();
console.log("Adjacency matrix:");
for (const row of a) console.log("  " + row.join(" "));
for (const [u, v] of edges) {
  assert(a[u][v] === 1 && a[v][u] === 1, `edge ${u}-${v} missing in matrix`);
}
for (let i = 0; i < 6; i++) {
  for (let j = 0; j < 6; j++) {
    assert(a[i][j] === a[j][i], "matrix must be symmetric");
  }
}
console.log("PASS: adjacency matrix matches edge list and is symmetric.");

const path = g.bfsShortestPath(0, 5);
console.log("BFS shortest path 0 -> 5:", JSON.stringify(path));
assert(path !== null && path.length - 1 === 3, "shortest path must use 3 edges");
for (let i = 0; i < path!.length - 1; i++) {
  assert(a[path![i]][path![i + 1]] === 1, "path must follow real edges");
}
assert(g.bfsShortestPath(0, 0)!.length === 1, "path to self is just [0]");
console.log("PASS: BFS found a valid 3-edge shortest path.");
console.log("ALL TYPESCRIPT GRAPH DEMOS PASSED");
