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
- `L`
  - lower triangular
- `U`
  - upper triangular matrix
  - Schur decomposition
  - contains eigenvalues on diagonal
  - A = PUP^{-1}
  - where P is orthogonal basis change matrix
- `V` - `eigenvector matrix`
  - is a basis
- $\Delta$ - `diagonal matrix`
- $\Lambda$ - `eigenvalues matrix`
- `J` (optimization)
- `S` - `similarity matrix`
- Gram matrix ? (do we include this - easy for things to spiral past specific one letter names)
- `H` - `Hessian matrix`

## Vectors

- `vectors`
  - each numerical value is referred to as a `coordinate`
- `tail` is at the origin
- `head` is at the coordinate
- addition, subtraction:
  - x + y = [(x_1 + y_1), ..., x_n + y_n ]
- scalar multiplication
  - ax = [ax_1, ..., ax_n]
- `dot product` aka `inner product`
  - $x \cdot y = \sum_{i=1}^{d} x_{i}y_{i} $
  - give you scalar projections
- `norm` or `euclidean norm`
  - $\|x\|^{2} = x \cdot x = \sum_{i}^{d}x_{i}^{2}  $
  - taking sqrt of this is `euclidean distance/length from origin`
- `cross products`
  - give you a new `vector`
- `orthogonal`
  - if `dot product` is 0 aka the angle bw is 90deg
- `orthonormal`
  - each pair of vectors is mutually orthogonal
  - norm of each vector is 1 (unit vectors)
- `projection`
  - a new set of coordinates wrt a new changed set of directions
  - "the 1-dim projection operation of a vector x on a unit vector is the dot product"
  - x is modified to point *in the direction of the unit vector*

### `Cauchy-Schwarz inqeuality`

The dot product between a pair of vectors is bounded *above* by the product of their lengths

$|\sum_{i=1}^{d}x_{i}y_{i}| = |\bar{x} \cdot \bar{y}  | \leq  \|\bar{x}\| \|\bar{y}\|  $

## Vectors and Matrices

- $A\bar{x}$ - where $\bar{x}$ is a col vector
  - remember ---| so this is each row times x
    - each rows products are added to create a scalar
    - produces nx1 column vector
- $\bar{y}A$ - where $\bar{y}$ is a row vector
  - remember ---|, so this is y times each col
    - produces 1xd row vector
- `linear transformation`
  - the Ax operation
  - transforms from a d-dimensional space to an n-dimensional space
    - nxd dx1 --> nx1
    - each row contains values that span the columns
  - Ax is effectively a **weighted sum** of the columns of A
    - so x is the weights, corresponding to each column
    - for each row, the weight is applied to the value for each column
  - $A\bar{x} = \sum_{i=1}^{d} x_{i}\bar{a}_{i} = \bar{b} $
    - each x_{i} corresponds to the weight of the ith direction a_{i}
      - which is the ith coordinate of b
    - scaling factor of directions. Each index of b corresponds to a direction

### Misc Properties

- `transpose`
  - $(AB)^{T} = B^{T} A^{T}$
  - swap order of terms within parenthesis
- `inverse`
  - sameas `non-singular`
  - $(AB)^{-1} = B^{-1}A^{-1}$ - need to invert the order when bring outside of the parentheses
    - (ABC)^{-1} = C^{-1}B^{-1}A^{-1}
  - An nxn square matrix A has linearly independent columns/rows if and only if it is invertible
    - if a square matrix with linearly independent, then invertible
  - the inverse of an `orthogonal matrix` is its transpose
  - if there are no 0 `eigenvalues` then a matrix must be invertible
  - the `determinant` of A det(A) != 0 if A is invertible
- `inverting singular matrices` - `matrix inversion lemma`
  - $(I + A)^{-1} = I - A + A^2 - A^3 + A^4 + ... + $ Infinite Terms
  - $(I - A)^{-1} = I + A + A^2 + A^3 + A^4 + ... + $ Infinite Terms
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
- `orthogonal`
  - $A^T A = I_d$
    - note, $PP^T = I$ is fine but P must be square, the above works for rectangular
  - $A^{T} = A^{-1}$
  - to make a new vector orthogonal to another vector:
    - first normalize v as $v/\|v\|$
    - $v\cdot u = 0 $
    - $v_{1}.u_{1} + v_{2}.u_{2} + v_{3}.u_{3} = 0  $
  - to make a new vector orthogonal to two other vectors:
    - $w = v \times u$ (cross product)
    - $w = \begin{bmatrix}i & j & k \\ v_{1} & v_{2} & v_{3} \\ u_{1} & u_{2} & u_{3}\end{bmatrix} $
    - $w = i|| - j|| + k||  $
  - new basis would be:
    - $P = [v, u, w]  $
- `orthonormal`
  - A set of vectors is `orthonormal` if each pair mutually orthogonal and the norm of each vector is 1
- `projection`
  - x = A^{T}b
  - here, we are projecting b onto the (orthonormal) columns of A to compute a new coordinate
- `nilpotent`
  - $A^{k} = 0 $ for some integer k
  - strictly triangular with diagonal entries of 0
- `idempotent`
  - V^{k} = V
- `indefinite`
  - symmetric matrices with both positive and negative eigenvalues
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

## Norms

- `Frobenius norm`
  - $\|A\|_{F} = \|A^{\top}\|_{F} = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{d} a_{ij}^{2}  }  $
- `energy`
  - Energy(A) = $\|A\|_{F}^{2} $ = Trace($AA^{T}$) = Trace($A^{T}A$)
- `trace`
  - tr(A) of a square matrix is defined by the sum of its diagonal entries
  - tr(A) = \sum_{i=1}^{n} a_{ii}^{2}
  - tr(A) is equal to the sum of the eigenvalues, whether it is diagonalizable or not
- Frobenius orthogonal
  - matrices A, B are Frobenius orthogonal when
  - tr(AB^T) = 0
  - or
  - $\|A + B\|_{F}^{2} = \|A\|_{F}^{2} + \|B\|_{F}^{2}  $

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

- `determinant`
  - the "volume" `scaling factor` of a matrix
  - product of `eigenvalues` $\det{(A)} \prod_{i} \lambda_{i}  $
- `eigenvectors` (right by default)
  - right $A\bar{x} = \lambda \bar{x} $
  - left $\bar{y}A = \lambda \bar{y} $
  - eigenvectors point in the directions that remain unchanged under transformation
  - they have a property where when A is multiplied against them, they result in the original vector just being scaled by some scalar value - the `eigenvalue` $\lambda$
  - solving for eigenvectors from eigevnalues:
    - solve (Q - \lambda_{1}I)v = 0
    - (do for each eigenvalue you have)
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
    - diagonal = all zeros except the diagonal
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

## `Raleigh quotient` or `Rayleigh quotient`

- Only works for 2nd order polynomials
- f(x) = x^T Q x - `quadratic form`
  - where x = [x1, x2, x3]
- start w/ polynomial
  - f(x_{1}, x_{2}, x_{3}) = 2x_{1}^{2} + 3x_{2}^{2} + 2x_{3}^{2} - 3x_{1}x_{2} - x_{2}x_{3} - 2x_{1}x_{3}
- **literally divide off-diagonal entries by 2** - ONLY EVER BY 2 BECAUSE ONLY EVER FOR 2ND ORDER BILINEAR TERM POLYNOMIALS
- write symmetric matrix of terms: as Q = 
  - [   2, -3/2,   -1]
  - [-3/2,    3, -1/2]
  - [  -1, -1/2,    2]
- knowing $\|x\| = 1$, means $x^T x = 1$
- min value occurs when x is the eigenvector of Q corresponding to the smallest eigenvalue
- solve the $2 - \lambda$ matrix to get the characteristic polynomial

Ok up next:

- $f(x) = \bar{x}^{T}Q\bar{x} = \bar{y}^{T}\Lambda \bar{y}  $
  - where: $\bar{y} = P^{T}\bar{x}  $
  - so $f(x) = (P^{T} \bar{x})^{T} \Lambda P^{T}\bar{x} $

Steps to solve for optimal solution:

- 1. Construct Q for $f(x) = \bar{x}^{T}Q\bar{x}$
  - diagonal is factors of {}x^2
  - off-diagonal are {}x_{1}x_{2} / 2 - always divide by 2
- 2. solve for eigenvalues with $det(Q - \lambda I) = 0$
- 3. solve for eigenvectors with $(Q - \lambda_{n} I)\bar{v}_{n} = 0$
  - plug in each eigenvalue \lambda_{n} corresponding to eigenvector \bar{v}_{n}
  - do gaussian elimination thing
  - normalize each vector
  - P = the normalized eigenvectors
- 4. $f(x) = (P^{T} \bar{x})^{T} \Lambda P^{T}\bar{x} $
  - or $\bar{y} = P^{T}\bar{x}  $
  - so $f(x) = \bar{y}^{T}\Lambda \bar{y}$
- 5. re-write $f(x_{1}, x_{2}) = f(y_{1}, y_{2}) = \lambda_{1}y_{1}^{2} + \lambda_{2}y_{2}^{2}
  - (where \lambda_{n} are the eigenvalues)
- 6. optimal solution occurs when *x is the eigenvector* that corresponds to the **smallest eigenvalue**

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

### 3x3 Rotation Matrices

$\begin{bmatrix}1 & 0 & 0 \\ 0 & \cos(\theta) & -\sin(\theta) \\ 0 & \sin(\theta) & \cos(\theta)\end{bmatrix} $

Here, the eigenvalues of all 3x3 rotation matrices are:

$[1, e^{i\theta}, e^{-i\theta} ]  $

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

"add c x (row 2) to row 1" - pre multiply

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
