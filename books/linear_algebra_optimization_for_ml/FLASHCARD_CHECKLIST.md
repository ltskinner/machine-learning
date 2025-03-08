# Flashcards

## Matrix Properties

| Property | Definition | Key Implication |
| - | - | - |
| Determinant | det(A) is a scalar that determines *invertibility* | det(A) = 0 means A is singular (non-invertible), at least one eigenvalue is 0 |
| Invertibility | $AA^{-1} = A^{-1}A = I $ | det(A) != 0, has full rank, has no zero eigenvalues, Ax = b has unique solutions for any b |
| Singularity | det(A) = 0, not invertible | at least one eigenvalue is 0 |
| Eigenvalues $\lambda$ | Scalars satisfying $A\bar{v} = \lambda\bar{v} $ | The scaling factors that describe how much the matrix stretches or compresses along eigenvector directions. Determine stability, transformations, and diagonalizability |
| Eigenvectors $\bar{v}$ | Nonzero vectors satisfying $A\bar{v} = \lambda\bar{v} $ | Point in the directions that remain unchanged under transformation. Form a basis if the matrix is diagonalizable |
|  |  |  |
| Rank | Number of linearly independent rows/columns | Full-rank **square** matrices are invertible |
| Trace | Tr(A) = sum of diagonal elements = sum of eigenvalues | Somehow this works regardless of if the matrix has been diagonalized |
| Norm | Measures the "size" of a matrix | Ex: Euclidean norm, Frobenius norm, spectral norm |
|  |  |  |
| Definiteness | Positive/negative definite, semidefinite, indefinite | Determines stability |
|  |  |  |
| Algebraic Multiplicity | Number of times an eigenvalue appears in the characteristic polynomial | Always >= GM |
| Geometric Multiplicity | Number of linearly independent eigenvectors for an eigenvalue | GM = AM for diagonalizability |
| Diagonalizability | $A = V \Delta V^{-1} $ | Simplifies computations in this form |
| Jordan Form | Generalization of diagonalization (for non-diagonalizable matrices) | Simplifies computations, reveals GM vs AM and how eigenvectors fail to form a complete basis |
|  |  |  |
| Sparsity | Most entries are zero | Important for computational efficiency |
|  |  |  |

## Matrix Types

| Matrix Type | Definition | Key Properties |
| - | - | - |
| Identity | Matrix with all ones on the diagonal |  |
| Diagonal | Diagonal elements are nonzero, all others are zero | Always diagonalizable, det = product of diagonal entries |
| Nilpotent | $A^{k} = 0 $ for some k | All eigenvalues are 0, not diagonalizable (unless zero matrix) |
| Idempotent | $A^{2} = A $ | Eigenvalues are only 0 or 1 |
| Projection | $A^{2} = A$, represents a transformation onto a subspace | Idempotent. Eigenvalues are only 0 or 1 |
| Orthogonal | $A^{T} = A^{-1} so $ $A^{T}A = I $ - columns/rows are orthonormal | Preserves lengths and angles, easy to invert, numerically stable, and eigenvalues are either 1, -1 |
| Unitary | $A^{*}A = I$ (complex analog of Orthogonal) | Preserves vector norms, eigenvalues lie on the unit circle |
| Symmetric | $A = A^{T} $ | Guarantees real eigenvalues, and possible diagonalizability (if matrix is real) |
| Hermitian | $A = A^{*}$ (comples analog of Symmetric) | Eigenvalues are real |
| Skew-Symmetric | $A^{T} = -A$ | Eigenvalues are either 0 or purely imaginary |
| Skew-Hermitian | $A^{*} = -A$ (complex analog of Skew-Symmetric) | Eigenvalues are purely imaginary |
|  |  |  |
| Positive Definite | $\bar{x}A\bar{x} > 0 $ for all x != 0 | All eigenvalues > 0, guaranteed invertible |
| Positive Semidefinite | $\bar{x}A\bar{x} >= 0 $ for all x | Eigenvalues >= 0, may be singular |
| Indefinite | $\bar{x}A\bar{x} $ can be positive or negative depending on x | Has both positive and negative eigenvalues |
| Negative Semidefinite | $\bar{x}A\bar{x} <= 0 $ for all x | Eigenvalues <= 0, may be singular |
| Negative Definite | $\bar{x}A\bar{x} < 0 $ for all x != 0 | All eigenvalues < 0, guaranteed invertible |
|  |  |  |
| Triangular (Upper/Lower) | All elements on and above/below diagonal are non-zero | Det = product of diagonal entries |
| Hessenberg | Triangular-like but with one extra non-zero sub/superdiagonal | Common in numerical algorithms |
|  |  |  |
| Stochastic | Nonnegative entries, rows sum to 1 | Common in Markov processes |
| Perron-Frobenius | A positive square matrix where the largest eigenvalue is real and positive | Used in probability and graph theory (e.g., Google PageRank) |
|  |  |  |
