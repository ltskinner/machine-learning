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

## Properties

- `orthogonal matrix`
  - `Q`
  - A matrix where its transpose is its inverse
  - AA^T = AA^{-1} = I
- `Givens rotations`
- `Householder reflection matrix`
  - `orthogonal matrix` that reflects a vector x into any "mirror" hyperplane of arbitrary orientation
  - v **must** be normalized
  - $Q = (I - 2\bar{v}\bar{v}^{\top})$ is an `elementary reflection matrix`

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
