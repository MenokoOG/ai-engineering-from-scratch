# 11 — Singular Value Decomposition
> Every matrix is just a rotation, a stretch, and another rotation.

**Project:** Used numpy's SVD to verify the rotate-scale-rotate picture of any matrix, compressed a 32x32 "image" with low-rank approximation, proved PCA-via-SVD equals PCA-via-covariance-eigendecomposition, and predicted a hidden movie rating with truncated SVD.

## What I built
- SVD of a small matrix with checks: U and V are orthogonal (pure rotations/reflections), and U @ diag(s) @ V^T rebuilds A exactly
- Step-by-step rotate -> scale -> rotate applied to a vector, matching A @ v
- Low-rank approximation of a 32x32 pattern: error and "energy kept" vs rank table
- PCA two ways: eigendecomposition of the covariance vs SVD of the centered data — same variances, same components
- Movie-ratings completion: hide one rating, subtract user means, iterate rank-2 truncated SVD to fill the hole

## Main points learned
- Any matrix, even a rectangular one, factors into rotate-scale-rotate. Eigendecomposition can't promise that; SVD always can.
- Singular values are the stretch factors, sorted biggest first. They tell you which directions matter.
- Low-rank approximation keeps the top-k stretches and throws the rest away — the best possible rank-k copy of the matrix.
- A few singular values often carry almost all the "energy": rank 4 of 32 kept 99.5% here.
- PCA via SVD of centered data gives identical answers to covariance eigendecomposition, but avoids forming X^T X, which squares the condition number and loses precision. That's why sklearn does it this way.
- Ratings matrices are approximately low-rank because people have a few taste types. That structure lets SVD guess missing entries.
- Eigenvector/singular-vector signs are arbitrary; compare results up to a sign flip.

## The algorithms, explained simply
**SVD.** Any matrix's action on a vector is: turn it (V^T), stretch it along the axes (Sigma), turn it again (U). Like a pasta machine: feed dough in at an angle, the rollers stretch it by fixed amounts, and it comes out turned another way.

**Low-rank approximation.** Keep only the biggest stretch directions and rebuild. Like a JPEG-style summary: store the broad strokes of the picture, drop the fine grain, and pay far fewer numbers for almost the same image.

**PCA via SVD.** Instead of building the covariance matrix and cracking it open, run SVD straight on the centered data table. Same answer, less numerical damage — like measuring a board directly instead of measuring its shadow twice.

**Matrix completion.** Assume the full ratings table is simple (few taste types), fill the hole with a bland guess, then repeatedly ask a rank-2 SVD "what value would make this table simple?" Like a Sudoku-style fill: the structure of everything else pins down the missing cell.

## How this shows up in AI
- LoRA fine-tunes LLMs by learning a low-rank update to weight matrices — pure truncated-SVD thinking.
- Recommendation engines (the Netflix Prize lineage) are truncated SVD on user-item matrices at scale.
- Model compression and attention-matrix analysis both lean on "most energy lives in a few singular values."
- sklearn's PCA — used constantly on embeddings — is implemented via SVD for speed and numerical stability.

## Run it
```
cd /home/claude/my-builds/phase-01-math-foundations/11-singular-value-decomposition
python3 project.py
```
