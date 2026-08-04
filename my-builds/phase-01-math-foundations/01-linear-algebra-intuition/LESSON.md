# 01 — Linear Algebra Intuition

> Vectors are lists of numbers that point somewhere; meaning lives in the direction.

**Project:** A mini vector toolkit built from plain Python lists (dot product, cosine similarity, rank, independence check), verified against numpy, plus a tiny LoRA-style low-rank demo. A TypeScript version covers the vector basics.

## What I built
- Dot product, norm (length), and cosine similarity from scratch
- Toy word embeddings where cosine similarity acts as "meaning similarity"
- Linear independence checker built on my own Gaussian-elimination rank function
- Matrix rank, checked against numpy
- LoRA-style demo: a big matrix W rebuilt exactly from two skinny matrices A and B
- project.ts: dot, norm, cosine on the same toy embeddings

## Main points learned
- A vector is just a list of numbers, but you can treat it as an arrow with a direction and a length.
- The dot product measures how much two vectors point the same way.
- Cosine similarity is the dot product with lengths removed. It only cares about direction.
- An embedding is a vector that stands for a word or item. Similar meanings get similar directions.
- Vectors are linearly independent when none of them can be built by mixing the others.
- Rank counts how many truly different directions a matrix contains.
- A low-rank matrix can be stored as two small matrices. That is the trick behind LoRA.

## The algorithms, explained simply
**Dot product.** Multiply matching entries and add them up. Think of it as a compatibility score: two arrows pointing the same way score high, perpendicular arrows score zero, opposite arrows score negative.

**Cosine similarity.** Divide the dot product by both lengths. It is like comparing two people's tastes by percentages instead of totals, so a loud voice and a quiet voice saying the same thing count as identical.

**Rank via Gaussian elimination.** Clean up the rows one at a time, subtracting out anything already covered by earlier rows. Rows that end up all zeros were just echoes of other rows. The count of surviving rows is the rank.

**Low-rank decomposition (LoRA idea).** If a big grid of numbers has only a few real patterns in it, you can store it as two skinny grids multiplied together. Like storing a checkerboard as "one row pattern times one column pattern" instead of every square.

## How this shows up in AI
- LLMs turn every token into an embedding vector; nearby directions mean related meanings.
- Semantic search and RAG rank documents by cosine similarity between embeddings.
- Attention scores inside transformers are basically dot products between query and key vectors.
- LoRA fine-tunes huge models cheaply by learning only a small A@B update instead of the whole weight matrix.

## Run it
```
python3 project.py
npx tsx project.ts
```
