# Dictionary - Cleaned up and more dense

## Named Matrices

- `A` - regular
- `B` - regular
- `D` - `data matrix`
  - typically nxd
    - each row n is a different sample
    - each col d is a feature
- `I` - `identity matrix`
- `P` - `basis matrix` - `projection matrix`
  - basis of rotation or projection
- `Q`
  - `orthogonal matrix`
  - QR decomposition
  - A matrix where its transpose is its inverse
  - AA^T = AA^{-1} = I
- `R`
  - upper triangular
- L
- `U`
  - Schur decomposition
  - upper triangular
  - contains eigenvalues on diagonal
  - A = PUP^{-1}
  - where P is orthogonal basis change matrix
- `V` - `eigenvector matrix`
  - is a basis
- $\Delta$ - `diagonal matrix`
- \Lambda - `eigenvalues matrix`
- J (optimization)
- `S` - `similarity matrix`
- Gram matrix ? (do we include this - easy for things to spiral past specific one letter names)
- `H` - `Hessian matrix`

### Misc Properties

- `orthonormal`
  - A set of vectors is `orthonormal` if each pair in the set is mutually orthogonal and the norm of each vector is 1.
- `symmetric matrix`
  - is a square matrix that is its own transpose
  - A = A^T
  - `spectral theorem` (describing symmetric matrices)
    - symmetric matrices are always `diagonalizable`
      - with real `eigenvalues`
    - have `orthonormal` eigenvectors
    - A can be diagonalized in the form:
      - $A = V\Delta V^{T}  $
      - with orthogonal matrix V
      - $A = V\Delta V^{T}  $ instead of $A = V\Delta V^{-1}  $

## Givens and Householder

- `Givens rotations`
  - of form G(plane_1, plane_2, angle)
  - take standard rotation matrix, and overlay on top of identity matrix
  - where rows and columns correspond to plane_1 and plane_2, with cos and sin operators
  - note, row and col G_{r|c} matrices are inverts of each other
  - there is usually an elementary reflection included as well
- `Householder reflection matrix`
  - `orthogonal matrix` that reflects a vector x into any "mirror" hyperplane of arbitrary orientation
  - v **must** be normalized
  - $Q = (I - 2\bar{v}\bar{v}^{\top})$ is an `elementary reflection matrix`

## EigenX

- `inverse`
  - $(AB)^{-1} = B^{-1}A^{-1}$ - need to invert the order when bring outside of the parentheses
    - (ABC)^{-1} = C^{-1}B^{-1}A^{-1}
  - An nxn square matrix A has linearly independent columns/rows if and only if it is invertible
    - if a square matrix with linearly independent, then invertible
  - the inverse of an `orthogonal matrix` is its transpose
  - if there are no 0 `eigenvalues` then a matrix must be invertible
  - the `determinant` of A det(A) != 0 if A is invertible
- `determinant`
  - the "volume" `scaling factor` of a matrix
  - product of `eigenvalues` $\det{(A)} \prod_{i} \lambda_{i}  $
- `eigenvectors`
  $A\bar{x} = \lambda A $
  - eigenvectors point in the directions that remain unchanged under transformation
  - they have a property where when A is multiplied against them, they result in the original vector just being scaled by some scalar value - the `eigenvalue` $\lambda$
- `eigenvalues`
  - the value an eigenvector is scaled by when A is multiplied against it
  - --> sum is Trace
  - --> product is Determinant
- `diagonalization`
  - $B = V\Delta V^{-1}  $
  - Matrices containing repeated eigenvalues and missing eigenvectors of the repeated eigenvalues are *not diagonalizable*

## Permutation Matrices

### Rotation

rotate ccw by \theta

$
\begin{bmatrix}
\cos(\theta) & -sin(\theta) \\
\sin(\theta) & cos(\theta)
\end{bmatrix}
$

DV_{r} will rotate each **row** of D

$ V_{r} =
\begin{bmatrix}
\cos{\theta} & \sin{\theta} \\
-\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

V_{c}D^{T} will rotate each **column** of D

$ V_{c} =
\begin{bmatrix}
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

### Reflection

reflect across X-axis

$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$

### Scale

scale x and y by factors of c_{1} and c_{2}

$
\begin{bmatrix}
c_{1} & 0 \\
0 & c_{2}
\end{bmatrix}
$

### Interchange

"interchange rows 1, 2" - pre multiply

$
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$

"interchange columns 1, 2" - post multiply

$
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$

### Addition

"add c x(row 2) to row 1" - pre multiply

$
\begin{bmatrix}
1 & c & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$

"Add c x (col 2) to col 1" - post multiply

$
\begin{bmatrix}
1 & 0 & 0 \\
c & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$

### Scaling

"multiply row 2 by c" - pre multiply

$
\begin{bmatrix}
1 & 0 & 0
0 & c & 0
0 & 0 & 1
\end{bmatrix}
$

"Multiply col 2 by c" - post multiply

$
\begin{bmatrix}
1 & 0 & 0 \\
0 & c & 0 \\
0 & 0 & 1
\end{bmatrix}
$
