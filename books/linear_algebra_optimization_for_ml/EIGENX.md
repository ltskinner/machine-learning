# EigenX

Diagonaliable matrix

- `diagonalizable matrix`
  - $A = V \Delta V^{-1} $
  - simultaneous scaling along d different directions (linearly independent)
    - $V $ = `eigenvectors`
      - if orthonormal, $V^{-1} = V^{T} $ and A is symmetric
  - d scale factors
    - $\Delta $ `eigenvalues`

Interpreting $A\bar{x} $

- 1. $V^{-1}\bar{x}$
  - computes coordinates of $\bar{x} $ in a basis system corresponding to the `eigenvectors` in V
- 2. $\Delta V^{-1}\bar{x} $
  - dilates the coordinates with `eigenvalue` scale factors in $\Delta $
- 3. $V \Delta V^{-1}\bar{x} $
  - transforms coordinates back to original basis system

Diagonalizable matrix A represents a `linear transformation` corresponding to anisotropic scaling in d linearly independent directions

## Determinants

Properties:

- Switching two rows (or columns) of a matrix A flips the sign of the det
- det(A) = det(A^T)
- A matrix with two identical rows (or cols) has a det of 0
- Row interchange operations do not change volume
- multiplying a single row of A with c to create A' multiplys the det(A) by factor of c
  - det(A') = c det(A)
  - multiplying entire matrix by c scales determinant by c^d
- det of A = 0 means non-invertible (singular)
- det of A is != 0 if matrix is invertible (non-singular)
  - parallelepiped of linearly independent vectors lies in a lower dimensional plane with 0 volume
- det(A^{-1}) = 1/det(A)
- multiplying A with diagonal matrix with values $\lambda_{1}...\lambda_{d} $ on the diagonal scales the volume of A by $\lambda_{1}...\lambda_{d} $
- `rotating` doesnt change det
- `reflecting` changes sign

## Eigenvectors and Eigenvalues

when $A\bar{x} = \lambda \bar{x} $

- $\bar{x} $ is an `eigenvector`
- $\lambda $ is its `eigenvalue`

![alt-text](./3_3_eigenvectors_eigenvalues.PNG)

when $AV = \Lambda V $

- $V $ is d linearly independent eigenvectors
  - also known as `basis change matrix`
- $\Lambda $ is d eigenvalues

becomes $A = V \Lambda V^{-1} $

also written as $A = V \Delta V^{-1}  $

- because $\Delta $ is a diagonal matrix with `eigenvalues` on the diagonal

### To Solve

To find `eigenvalues`:

- $\det{(A - \lambda I)} = 0 $
  - the `characteristic equation`/`characteristic polynomial`
- $\det{(A - \lambda I)} = (\lambda_{1} - \lambda)(\lambda_{2} - \lambda).. $
  - standard to order in ascending: 1, 3, 5, ...

To find `eigenvectors`:

- $(A - \lambda_{i} I)v_{i} = 0  $
  - plug in each `eigenvalue` lambda and solve for v_{i}
  - v_{i} is the corresponding `eigenvector`
  - note each $v_{i} $ constitutes $V = [v_{i}, ...] $
  - it is common (but not required) to normalize each v_{i}
    - v_{i} / \|v_{i}\| (sqrt(v^2))
  - to `diagonalize`
    - $A = V \Delta V^{-1} $
      - $v^{-1} = 1/det[d -b] $
      -                [-c a]
      - only `linearly independent` if **NO** repeated `eigenvalues`
      - if **NOT** `linearly independent`, **CANNOT** be `diagonalized`
- `right eigenvector` by default
  - span the `column space`
  - `col` *vectors* external to A
- `left eigenvector`
  - span the `row space`
  - `row` *vectors* external to A
  - $\bar{y}A = \lambda \bar{x}  $
  - if A is `symmetric`
    - the left and right eigenvectors are transpositions of each other
    - else, they are different
  - span the `row space`
- $V$
  - contains (right) `eigenvectors`
  - to find:
    - subsitute eigenvalues into A
    - set A' = 0 and solve for variables
    - remember, free variables can just be 1
- $V^{-1} $
  - contains (left) `eigenvectors`
  - $V^{-1} = \frac{1}{\det{(ad-bc)}} \begin{bmatrix}d & -b\\-c & a\end{bmatrix} $

### Complex Eigenvalues Section 3.3.1

Complex eigenvalues of a real matrix must:

- occur in conjugate pairs of the form:
  - a + bi
  - a - bi
- corresponding eigenvectors occur in pairs:
  - \bar{p} + i\bar{q}
  - \bar{p} - i\bar{q}

### Algebraic Multiplicity

- `algebraic multiplicity`
  - the number of times an `eigenvector` occurs in the `characteristic polynomial`
  - when each `eigenvector` occurs once, the `algebraic multiplicity` = 1
    - this means it is `diagonalizable`
  - when > 1:
    - case 1:
      - infinite number of `diagonalizations` exist
      - think of I matrix - so many different ways to construct V, where \Delta is I
    - case 2:
      - no `diagonalization` exists
      - when number of `linearly independent` `eigenvectors` is less than `algebraic multiplicity`
      - matrices containing *repeated* `eigenvalues` and *missing* `eigenvectors` of the *repeated* `eigenvalues` are **NOT** `diagonalizable`
  - when < 1:
    - `non-diagonalizable` aka `non diagonalizable` aka `not diagonalizable`
    - `diagonalization` does not exist
      - `non-diagonalizable` matrices contain a "residual" rotation
      - `polar decomposition` - when we include the "residual" rotation matrix
        - rotation matrices are `diagonalizable`, but have **complex** eigenvalues
    - closest we can get to `diagonalization` is `Jordan Normal Form` which contains `Jordan chains` aka `defective`

### Characteristic Polynomial

- $(A - \lambda_{i})^{k} $ - an eigenvalue \lambda with multiplity k
  - (eigenvalue repeats k times)
  - null space has dimensionality k

### Jordan Normal Form aka Jordan Chains

Take: $A = V U V^{-1} $, where U is `upper triangular` and is "almost diagonal"

$V = \bar{v}_{1}...\bar{v}_{m(i)} $

Where:

- $\bar{v}_{1} $ is an `ordinary eigenvector`
  - satisfies $A\bar{v}_{1} = \lambda\bar{v}_{1} $
  - for the real eigenvectors
- $...\bar{v}_{m(i)} $ are `generalized eigenvectors`
  - satisfies $A\bar{v}_{j} = \lambda\bar{v}_{j} + \bar{v}_{j-1} $ for j > 1
  - here, $\bar{v}_{j-1} = \bar{v}_{1} $ when j = 2 (the original \bar{v}_{1})
- obtained as:
  - $\bar{v}_{m(i) - r} = (A - \lambda I)^{r}\bar{v}_{m(i)} $
  - for each r from 1 to m(i) - 1
  - following form of:
    - $(A - \lambda I)^{k}\bar{x = 0} $ and there are r < k solutions

Remember - this "almost" diagonalizes A as U, and has some rules

- at most (d - 1) entries above the diagonal can be 1 or 0
  - is 0 IFF corresponding eigenvector is orinary
  - is 1 IFF not ordinary (is generalized)
  - entries above the diagonal called a `super-diagonal entry`
- usually contain a small number of `super-diagonal` 1s
- triangularizations are not unique
  - different triangularizations come from imposing different constaints on the basis vectors and triangular matrix
- also possible for eigenvectors and eigenvalues to be complex, even for real matrices

### Similar Matrix Families - *Sharing Eigenvalues*

- `similar matrices`
  - A and B are similar when $B = VAV^{-1} $ (or vice versa)
  - if we multiply a vector by A or B, the same `transformation` occurs
    - (as long as the basis is appropriately chosen)
    - e.g. two similar matrices perform a 60deg rotation, but axis of rotation is deferent
    - e.g. two similar transforms scale a vector by the same factor, but in different directions
  - similar matrices have the same `eigenvalues` (and corresponding multiplicities)
    - `eigenvectors` not the same
  - the `traces` of similar matrices are equal, and equal to sum of eigenvalues of that family

#### Jordan Normal Forms of Similar Matrices

- A and B are similar matrices where $B = VAV^{-1} $
- their jordan normal forms are related as:
  - $A = V_{1} U V_{1}^{-1} $
  - $B = V_{2} U V_{2}^{-1} $
  - where $V_{2} = V V_{1}  $
