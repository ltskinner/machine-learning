# Dictionary

Random stuff with no better home

- regularization
  - A technique to prevent overfitting by adding a penalty term (like L1 or L2) to the loss function, discouraging overly complex models.
  - controls model complexity
- normalization
  - A preprocessing step to scale input features to a common range (e.g. [0, 1] or mean 0 and variance 1 aka 'standard normal' distribution)
  - improves model convergence and performance, and ensures features are on comparable scales
- monotonic
  - a function that is always only increasing or only decreasing over its domain
- semantics
  - in text
    - the meaning of words, sentences, and their relationships in a sentence
  - in images
    - the objects, scene, and their relationships in the visual space

## P and NP

| Category | Definition | Solution Time | Verification Time | Examples | Relationship to Others |
| - | - | - | - | - | - |
| P (Polynomial Time) | Problems that can be solved efficiently (in polynomial time) | Polynomial time - O(n^k) | Polynomial Time | Sorting, Graph Search (BFS, DFS) | P \subseteq NP |
| NP (Nondeterministic Polynomial Time) | Problems where a given solution can be verified efficiently | Unknown, but may be exponential | Polynomial Time | Sudoku solution verification | Contains P; NP-complete problems are the hardest in NP |
| NP-Complete | Problems that are both in NP and NP-hard | Unknown, likely exponential | Polynomial Time | 3-SAT, Graph Coloring | Solving any NP-complete problem efficiently would imply P = NP |
| NP-Hard | Problems that are at least as hard as NP-complete problems, but not necesarily in NP | Unknown, possibly not even verifiable in polynomial time | May not be verifiable in polynomial time | Traveling Salesman Problem | If an NP-hard problem is in NP, its NP-complete |
