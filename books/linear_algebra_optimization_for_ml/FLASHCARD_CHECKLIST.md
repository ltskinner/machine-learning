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
| Orthonormality | rows/cols are orthonormal if: 1. unit length, 2. mutually orthogonal (perpendicular (90deg angles yes really - this differs from linear independence)) | you may have linearly independent vectors, however they are only orthogonal if they meet at 90deg angle |
|  |  |  |
| Algebraic Multiplicity | Number of times an eigenvalue appears in the characteristic polynomial | Always >= GM |
| Geometric Multiplicity | Number of linearly independent eigenvectors for an eigenvalue | AM = GM for diagonalizability |
| Diagonalizability | $A = V \Delta V^{-1} $ | Simplifies computations in this form |
| Jordan Form | Generalization of diagonalization (for non-diagonalizable matrices) | Simplifies computations, reveals GM vs AM and how eigenvectors fail to form a complete basis |
|  |  |  |
| Sparsity | Most entries are zero | Important for computational efficiency |
|  |  |  |
| Inconsistent | b $\notin$ column space of A | No solutions, but best fit is possible (trailing rows of 0s) |
| Consistent | b $\in$ column space of A | At least one solution, either unique or infinite (no trailing rows of 0s) |
|  |  |  |

## Matrix Types

| Matrix Type | Definition | Key Properties |
| - | - | - |
| Identity | Matrix with all ones on the diagonal |  |
| Diagonal | Diagonal elements are nonzero, all others are zero | Always diagonalizable, det = product of diagonal entries |
| Nilpotent | $A^{k} = 0 $ for some k | All eigenvalues are 0, not diagonalizable (unless zero matrix) |
| Idempotent | $A^{2} = A $ | Eigenvalues are only 0 or 1 |
| Orthogonal | $A^{T} = A^{-1} so $ $A^{T}A = I $ - columns/rows are orthonormal | Preserves lengths and angles, easy to invert, numerically stable, and eigenvalues are either 1, -1 |
| Unitary | $A^{*}A = I$ (complex analog of Orthogonal) | Preserves vector norms, eigenvalues lie on the unit circle |
| Symmetric | $A^{T} = A $ | Guarantees real eigenvalues, and possible diagonalizability (if matrix is real) |
| Hermitian | $A^{*} = A$ (comples analog of Symmetric) | Eigenvalues are real |
| Skew-Symmetric | $A^{T} = -A$ | Eigenvalues are either 0 or purely imaginary |
| Skew-Hermitian | $A^{*} = -A$ (complex analog of Skew-Symmetric) | Eigenvalues are purely imaginary |
|  |  |  |
| Positive Definite | $\bar{x}A\bar{x} > 0 $ for all x != 0 | All eigenvalues > 0, guaranteed invertible |
| Positive Semidefinite | $\bar{x}A\bar{x} >= 0 $ for all x | Eigenvalues >= 0, may be singular |
| Indefinite | $\bar{x}A\bar{x} $ can be positive or negative depending on x | Has both positive and negative eigenvalues |
| Negative Semidefinite | $\bar{x}A\bar{x} <= 0 $ for all x | Eigenvalues <= 0, may be singular |
| Negative Definite | $\bar{x}A\bar{x} < 0 $ for all x != 0 | All eigenvalues < 0, guaranteed invertible |
| Strictly Positive (Negative) | All individual values in matrix are > 0 (< 0) |  |
|  |  |  |
| Triangular (Upper/Lower) | All elements on and above/below diagonal are non-zero | Det = product of diagonal entries |
| Hessenberg | Triangular-like but with one extra non-zero sub/superdiagonal | Common in numerical algorithms |
|  |  |  |
| Stochastic | Nonnegative entries, rows sum to 1 | Common in Markov processes |
| Perron-Frobenius | A positive square matrix where the largest eigenvalue is real and positive | Used in probability and graph theory (e.g., Google PageRank) |
|  |  |  |

## Transformation Matrices

| Transformation | Definition | Key Properties |
| - | - | - |
| Permutation Matrix | Rearranges rows or cols | Square, orthogonal ($P^{T} = P^{-1}$), entries are 0 or 1 |
| Rotation Matrix | Rotates vectors around an axis | Orthogonal ($R^{T}R = I$), det(R) = +- 1 |
| Reflection Matrix | Flips points symmetrically across a plane or line | Symmetric ($A = A^{T} $), $\lambda_{i} = +- 1$ |
| Scaling Matrix | Stretches or compresses along principal axes | Diagonal form, $\lambda_{i}$ are scaling factors |
| Interchange Matrix | Swaps two rows or columns | Orthogonal ($P^{T} = P^{-1}, P^{T}P = I $) |
| Addition Matrix | Adds values of one row/col to another row/col | Invertible |
|  |  |  |
| Givens Rotation Matrix | 2d rotation matrix, performs localized rotation as G(row/col_1, row/col_2, angle) | Orthogonal ($P^{T} = P^{-1} $), used in QR decomp |
| Householder Reflection Matrix | Reflects vectors across hyperplane | Orthogonal ($P^{T} = P^{-1}, P^{T}P = I $) |
|  |  |  |
| Shear Matrix | Shifts one coordinate direction relative to another. Preserves straightness of lines, parallelness of lines, but distorts angles (breaking orthogonality b/c line intersection). Can stretch, shrink, or leave unchanged (per-dimension) | At least one $\lambda $ is 1 (b/c some vectors are unchanged) |
| Projection aka Basis (Matrix) | $A^{2} = A$, maps vectors onto a subspace (e.g. line or plane) | Idempotent. Eigenvalues are only 0 or 1. If there is an eigenvalue of 0, then it projects into a lower dimension |
| Skew-Symmetric Matrix | $A^{T} = -A $ Rotation-like behavior without preserving vector magnitude | Eigenvalues are 0 or purely imaginary |
|  |  |  |
| Translation Matrix | Moves points without changing orientation, shape, or scale - shifts all points a fixed amount in each direction | like "adding an offset", but translation is a multiplication operation, whereas offset is element-wise addition (neither of these are linear operations) |
| Affine Transformation Matrix | Combines mutliple xformations, such as rotation, scaling, translation |  |
| Similarity Transformation Matrix | Changes a matrix while preserving eigenvalues | $A' = P^{-1} A P $ |
| Fourier Transformation Matrix | Converts signals between time and frequency domains | Complex-valued, $\lambda $ are roots of unity |

## Operations

| Operation | Definition | Key Properties |
| - | - | - |
| Dot Product (Inner Product) | $a \cdot b = \sum a_{i}b_{i} $ | measures how aligned two vectors are. returns scalar |
| Outer Product | $ab^T$ = (nxd) rank-1 matrix (all rows/cols are multiples of each other, so multiples of a single vector) | Represents a projection or transformation |
| Cross Product | Produces a vector orthogonal to both inputs |  |
|  |  |  |
| Affine Transformation | Combination of a `linear transformation` with a `translation` |  |
| Linear Transformation (Linear Operator) | f(x) = Ax (so, b = 0, not translating) | maps between two vector spaces (maps origin of V to origin of W aka "preserves the origin") |
| Linear Function | f(x) = Ax + b (where b != 0 is the translation vector) | when `translation` is allowed, origins are not mapped to each other |
|  |  |  |

## Four Fundamental Subspaces of a Matrix

| Subspace | Span | Definition | Basis | Dimensionality | Identifies |
| - | - | - | - | - | - |
| Right Null Space | "null space of the row space" | Set of solution vectors x such that $A\bar{x} = 0$ | solutions to $A\bar{x} = 0 $ | N(A) = n - rank(A) | Identifies "lost inputs" in R^d (A annihilates input x by mapping to 0) |
| Row Space | Span of $A^{T}$ | Contains all possible linear combinations of row vectors | Independent rows of A | rank(A) | Identifies "constraints" on R^d |
| Column Space | Span of $A$ | Contains all possible linear combinations of column vectors | Independent cols of A | rank(A) | Identifies "reachable outputs in R^n" |
| Left Null Space | "null space of the column space" | Set of solution vectors y such that $A^{T}\bar{y} = 0$; vectors orthogonal to columns of A | solutions to $A^{T}\bar{y} = 0 $ | N(A) = d - rank(A) | Identifies "unreachable outputs" in R^n - directions that input x cannot reach b/c R^n fails to span |

## Solution Cases for $A\bar{x} = \bar{b} $

| Case | Condition | Geometric Interpretation | Implication on Rank |
| - | - | - | - |
| No Solution | b is not in col space, but orthogonal best fit solution is possible | inconsistent system (trailing rows of 0s in RREF), over-determined systems | # rows > # cols (tall) |
| Unique Solution | b is in col space, linearly independent cols | Solution: $\bar{x} = A^{-1}\bar{b} $ | square matrices |
| Infinite Solutions | b in col space, not linearly independent cols | Solution: has free parameters (free columns in RREF) | # cols > # rows (wide) |

## Scratchpad

- inverse
  - left-inverse
  - right inverse
- similarity matrix
- defective matrix (next to consistent/inconsistent)
  - not diagonalizable
  - `simultaneously diagonalizable`
  - `similar matrices`

- gram matrix

processes/operations:

- push through identity?
- QR decomp
- gram-schmidt orthogonalization
- LU decomopsition
- moore-penrose
  - form of projection matrix
  - symmetric, idempotent
- Neumann Series
- Sherman-Morrison-Woodburry Identity
- schur decomp
- gaussian elimination
  - row echelon form (RREF too)
  - back substitution
    - parametric form
- spectral theorem
- singular value decomposition (SVD)
- cayley-hamilton
- polynomial function
- raleigh quotient / rayley quotient
- cauchy-schwarz inequality

### eventually 2

- variance
- covariance
