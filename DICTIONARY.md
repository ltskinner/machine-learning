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

## Convex and Non-Convex Optimization Problems

### Convex

The feasible region (constraint set) is a convex set (e.g. a line segment connecting any two points in the set lies entirely within the set)

Characteristics:

- Single glboal minimum
  - no local minima other than the global minimum, making optimization easier
- Efficient algorithms
  - Solvers like Gradient Descent can converge reliably
- P complexity
  - Many convex problems have efficient solutions

Easier to optimize and guarantee global optimal solutions but may be limited in their expressiveness

### Non-Convex

Some parts of the constraint set may not be connected

Characteristics:

- Multiple local minima
  - many sub-optimal solutions that gradient-based methods can get stuck in
- Requires heuristics
  - methods like random restarts, simulated annealing, and evolutionary algorithms are often used

More expressive but challenging to optimize due to presence of multiple local optima
