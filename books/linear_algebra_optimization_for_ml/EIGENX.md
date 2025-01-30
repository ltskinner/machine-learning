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
