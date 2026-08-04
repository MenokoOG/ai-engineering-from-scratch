// Lesson 14 (TypeScript): norms, cosine similarity, and nearest-neighbor search.
// Run: npx tsx project.ts

function l1Norm(v: number[]): number {
  return v.reduce((s, x) => s + Math.abs(x), 0);
}

function l2Norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function dot(a: number[], b: number[]): number {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}

function cosineSim(a: number[], b: number[]): number {
  return dot(a, b) / (l2Norm(a) * l2Norm(b));
}

function euclidean(a: number[], b: number[]): number {
  return l2Norm(a.map((x, i) => x - b[i]));
}

function assertClose(actual: number, expected: number, label: string, tol = 1e-9): void {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`FAIL ${label}: got ${actual}, expected ${expected}`);
  }
  console.log(`  ok: ${label} = ${actual.toFixed(6)}`);
}

console.log("=== 1) norms on v = [3, -4, 1] ===");
const v = [3, -4, 1];
assertClose(l1Norm(v), 8, "L1 norm");
assertClose(l2Norm(v), Math.sqrt(26), "L2 norm");

console.log("\n=== 2) cosine similarity on toy embeddings ===");
// same toy embeddings as project.py: [animal-ness, pet-ness, machine-ness]
const emb: Record<string, number[]> = {
  cat: [2.0, 1.8, 0.1],
  kitten: [1.9, 2.0, 0.2],
  CAT_ESSAY: [20.0, 18.0, 1.0], // same direction as cat, 10x longer
  car: [0.2, 0.1, 2.0],
};
for (const other of ["kitten", "CAT_ESSAY", "car"]) {
  const e = euclidean(emb.cat, emb[other]);
  const c = cosineSim(emb.cat, emb[other]);
  console.log(`  cat vs ${other.padEnd(9)} euclid=${e.toFixed(3).padStart(7)}  cosine=${c.toFixed(3)}`);
}
if (!(cosineSim(emb.cat, emb.CAT_ESSAY) > 0.999)) throw new Error("cosine should ignore length");
if (!(euclidean(emb.cat, emb.CAT_ESSAY) > euclidean(emb.cat, emb.car))) {
  throw new Error("euclidean should be fooled by length");
}
console.log("  cosine sees cat and CAT_ESSAY as identical; raw distance does not");

console.log("\n=== 3) tiny nearest-neighbor search ===");
function nearestNeighbors(
  query: number[],
  db: Record<string, number[]>,
  k: number,
): Array<{ name: string; score: number }> {
  return Object.entries(db)
    .map(([name, vec]) => ({ name, score: cosineSim(query, vec) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

const docs: Record<string, number[]> = {
  "adopting a kitten": [1.8, 2.1, 0.1],
  "cat food review": [2.0, 1.5, 0.3],
  "engine repair 101": [0.1, 0.0, 2.2],
  "best road trip cars": [0.3, 0.2, 1.9],
  "why dogs purr (satire)": [1.5, 1.7, 0.2],
};
const query = emb.cat;
console.log("  query: 'cat' embedding, top 3 matches by cosine:");
const top = nearestNeighbors(query, docs, 3);
for (const { name, score } of top) {
  console.log(`    ${score.toFixed(3)}  ${name}`);
}
if (top[0].name.includes("engine") || top[0].name.includes("cars")) {
  throw new Error("nearest neighbor should be an animal doc");
}
if (!top.every((t) => !t.name.includes("engine"))) {
  throw new Error("engine doc should not be in top 3");
}
console.log("  animal docs win, machine docs lose: that is semantic search in miniature");

console.log("\nAll TypeScript checks passed.");
