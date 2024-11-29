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
  - upper triangular matrix
  - Schur decomposition
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
- `nilpotent`
  - when a matrix satisfies $A^{k} = 0 $ for some integer k
  - strictly upper triangular with diagonal entries of 0
- `idempotent`
  - V^{k} = V
- `idempotent property` of `projection matrices`:
  - $P^2 = P = (QQ^{T})(QQ^{T}) = Q(Q^{T}Q)Q^{T} = QQ^{T} $ - note (Q^{T}Q) is an identity
- `energy`
  - another name for the squared Frobenius norm
  - The energy of a rectangular matrix A is equal to the trace of either AA^{\top} or A^{\top}A
    - $\|A\|_{F}^{2} = Energy(A) = tr(AA^{\top}) = tr(A^{\top}A)  $
- `trace`
  - tr(A) of a square matrix is defined by the sum of its diagonal entries
  - tr(A) = \sum_{i=1}^{n} a_{ii}^{2}
  - tr(A) is equal to the sum of the eigenvalues, whether it is diagonalizable or not

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
- `eigenvectors` (right by default)
  - right $A\bar{x} = \lambda \bar{x} $
  - left $\bar{y}A = \lambda \bar{y} $
  - eigenvectors point in the directions that remain unchanged under transformation
  - they have a property where when A is multiplied against them, they result in the original vector just being scaled by some scalar value - the `eigenvalue` $\lambda$
- `eigenvalues`
  - the value an eigenvector is scaled by when A is multiplied against it
  - --> sum is Trace
  - --> product is Determinant
- `diagonalization`
  - $B = V\Delta V^{-1}  $
  - Matrices containing repeated eigenvalues and missing eigenvectors of the repeated eigenvalues are *not diagonalizable*
  - I think all symmetric, orthogonal matrices are diagonal matrices
    - symmetric = A = A^T
    - orthogonal = A^T = A^-1
    - so, A = A^-1
- `defective matrix`
  - a matrix where a `diagonalization` *does not exist*
  - missing eigendirections (eigenvectors) that contribute distinctly
- `simultaneously diagonalizable`
  - A diagonalizable matrix family that shares eigenvectors (but not eigenvalues)
  - the columns of V are the eigenvectors of both A and B
    - $A = V \Delta_{1} V^{T}  $ or $V^{T} A V = \Delta_{1}  $
    - $B = V \Delta_{2} V^{T}  $ or $V^{T} B V = \Delta_{2}  $
    - Here, \Delta_{1} and \Delta_{2} are diagonalizable matrices
- `similar matrices`:
  - when $B = VAV^{-1} $ for two matrices A and B
  - similar matrices have the same eigenvalues (and their corresponding multiplicities)
  - similar matrices perform similar operations, but in different basis systems

## Jordan Normal Form

For `defective` matrices: `Jordan Normal form` is the closest we can get to a `diagonalization`

$A = V U V^{-1} $

- U is an `upper triangular matrix`
- diagonal entries containing eigenvalues in the same order as the corresponding generalized eigenvectors in V
- entries above the diagonal can be 0 or 1.
  - at most (d-1) entries
  - is 0 if and only if the corresponding eigenvector is an ordinary eigenvector
  - is 1 if it is not an ordinary eigenvector

## Cayley-Hamilton

if you have a characteristic equation:

- p(\lambda) = (2 - \lambda)(2 - \lambda) - (1)(1)
- = \lambda^2 - 4\lambda + 3
- = A^2 - 4A + 3

--> you can just insert the matrix A into \lambda directly for the characteristic equation

"a matrix satisfies its own polynomial equation"

formal from book:

- the inverse of a `non-singular` matrix can always be expressed as:
  - a polynomial of degree (d-1)
- Let A be any matrix with characteristic polynomial $f(\lambda) = \det{(A - \lambda I)}  $.
- Then, f(A) evaluates to the zero matrix

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
