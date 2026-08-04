# my-builds — AI Engineering from Scratch, done by hand

My project builds for the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum. Every algorithm implemented from scratch first, verified against numpy/sklearn second.

## What's here

```
my-builds/
├── CHEATSHEET.md            ← master cheat sheet: every algorithm, pattern, and key fact
├── phase-01-math-foundations/
│   ├── 01-linear-algebra-intuition/
│   │   ├── project.py       ← my from-scratch build (runnable, self-verifying)
│   │   ├── project.ts       ← TypeScript version (where present)
│   │   ├── LESSON.md        ← what I built, what I learned, algorithms in plain English
│   │   └── FEEDBACK.md      ← errors/bugs found in the course's own materials
│   └── ... (22 lessons)
├── VERSION
└── CHANGELOG.md
```

## Run anything

```bash
cd phase-01-math-foundations/<lesson>
python3 project.py        # every file runs standalone, exits 0, asserts its own results
npx tsx project.ts        # where a TypeScript version exists (01, 05, 14, 19, 21)
```

Dependencies: `pip install -r requirements.txt` (numpy, scikit-learn). TypeScript files are dependency-free.

## Status

| Phase | Lessons built | Verified |
|---|---|---|
| 01 — Math Foundations | 22 / 22 | ✅ all run clean |

Next up: Phase 2 — ML Fundamentals.

---

*classHuman AI — driven by LAHA (Love All Humans Always).*
