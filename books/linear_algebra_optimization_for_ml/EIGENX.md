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
