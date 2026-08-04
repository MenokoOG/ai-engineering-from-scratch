// Lesson 01: vector basics in TypeScript (dependency-free).

function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("length mismatch");
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

function cosineSimilarity(a: number[], b: number[]): number {
  return dot(a, b) / (norm(a) * norm(b));
}

function assertClose(actual: number, expected: number, label: string): void {
  if (Math.abs(actual - expected) > 1e-9) {
    throw new Error(`${label}: got ${actual}, expected ${expected}`);
  }
  console.log(`  ok: ${label} = ${actual.toFixed(4)}`);
}

console.log("=== Vector basics (TypeScript) ===");
assertClose(dot([1, 2, 3], [4, 5, 6]), 32, "dot([1,2,3],[4,5,6])");
assertClose(norm([3, 4]), 5, "norm([3,4])");
assertClose(cosineSimilarity([1, 0], [0, 1]), 0, "cosine of perpendicular vectors");
assertClose(cosineSimilarity([2, 0], [7, 0]), 1, "cosine of same-direction vectors");
console.log();

console.log("=== Toy word embeddings: cosine = meaning similarity ===");
// dims: [animal-ness, size, royalty, food-ness]
const emb: Record<string, number[]> = {
  cat:   [0.9, 0.2, 0.0, 0.1],
  dog:   [0.9, 0.3, 0.0, 0.1],
  king:  [0.1, 0.5, 0.9, 0.0],
  queen: [0.1, 0.4, 0.9, 0.0],
  pizza: [0.0, 0.2, 0.0, 0.9],
};

const pairs: Array<[string, string]> = [
  ["cat", "dog"], ["king", "queen"], ["cat", "pizza"], ["king", "pizza"],
];
for (const [w1, w2] of pairs) {
  console.log(`  sim(${w1}, ${w2}) = ${cosineSimilarity(emb[w1], emb[w2]).toFixed(3)}`);
}

if (cosineSimilarity(emb.cat, emb.dog) <= cosineSimilarity(emb.cat, emb.pizza)) {
  throw new Error("expected cat~dog to beat cat~pizza");
}
console.log("  -> related words score higher, as expected");
console.log();
console.log("All checks passed.");
