# QR Decomposition and LR Decomposition

## Comparing QR and LU Decomposition

### Purpose

- QR Decomposition
  - works for square and rectangular matrices
  - to `orthogonalize` a matrix
  - especially in
    - solving least squares problems (over-determined systems)
    - eigenvalue computations
  - typically more numerically stable
- LU Decomposition
  - typically faster
  - applies only to square matrices
  - to `factorize` a square matrix
  - for
    - efficient solving of linear systems (Ax = b)
    - computing `determinants`

## LU Decomposition

- $A = LU$
  - $a_{11} = l_{11}u_{11}$
- Express matrix as product of
  - a (square) lower triangular matrix L
    - catalogues the operations made during `Gaussian Elimination`
  - a (rectangular) upper triangular matrix U
    - the output of `Gaussian Elimination`
- (wait this is nasty)
  - row addition operations are always lower triangular `L` in GE
  - row interchange operations is a permutation matrix `P`
    - these steps can be expressed as
    - $PL_{m}L_{m-1}...L_{1}A = U  $
    - $A = L_{m}^{-1}L_{m-1}^{-1}...L_{1}^{-1}P^{-1}U  $
    - $A = L_{m}^{-1}L_{m-1}^{-1}...L_{1}^{-1}P^{T}U  $
    - $A = LP^{T}U  $ = $A = P^{T}LU  $
- final form:
  - $PA = LU  $
- can use to invert a matrix
  - basically perform row operations on an identity matrix until reaching the original matrix
  - stacking all these operations is the inversion
  - "a sequence of row operations that transforms A to the identity matrix will transform the identity matrix B = A^{-1}"

### Cholesky Factorization: Symmtric LU Decomposition

- Remember, we want A = BB^T
  - A = B(PP^T)B^T = (BP)(BP)^T
- Cholesky Factorization = LL^T
  - where L is some lower-triangular matrix
  - $L = R^T $ from `QR decomposition`
    - because the transpose of an upper triangular = lower triangular
- can only be used for positive definite matrices
- Cholesky factorizations are unique
- faster to compute this than generic LU
- preferred approach for testing the positive definiteness of a matrix
  - b/c if not pd:
    - then a 0 will occur on diagonal
    - or, will be forced to compute sqrt of a negative value

$a_{ij} = \sum_{k=1}^{d} l_{ik}l_{jk} = \sum_{k=1}^{j}l_{ik}l_{jk} L  $

Where the first sum is $A_{ij} = (LL^T)_{ij}  $

The first column of L can be computed by setting j=1, and iterating all i>=j:

- $l_{11} = \sqrt(a_{11}) $
- $l_{i1} = a_{i1} / l_{11} \forall i > 1 $

The second column:

- $l_{22} = \sqrt(a_{22} - l_{21}^{2}) $
- $l_{i2} = (a_{i2} - l_{i1}l_{21})/l_{22} \forall i > 2 $

Generalized routine:

```txt
Initialize L = [0]_{dxd}
for j = 1 to d do
  l_{jj} = sqrt(a_{jj} - \sum_{k=1}^{j-1} l_{jk}^{2});
  for i = j + 1 to d do
    l_{ij} = (a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk} / l_{jj})
  endfor
endfor
return L = [l_{ij}]
```

## Gram-Schmidt Process / QR Decomposition

Purpose: transform a set of `linearly independent vectors` into an orthogonal (or orthonormal) set of vectors

In QR decomposition, this is applied to the *columns* of A, where:

- Q is an orthogonal (or orthonormal) matrix whose *columns are orthogonal* `unit vectors` (orthonormal)
  - Q is produced using `Gram-Schmidt`
- R is an upper triangular matrix

We decompose A into A = QR

The projection matrix P = QQ^T

One can compute the projection of a vector as: QQ^T b = b'

The best-fit solution x to Ax = b, as: Rx = Q^{T}b' = Q^{T}b

Which you can solve using standard back subsitition into R as R|b >> x

For:

$
\begin{bmatrix}
a_11 & a_12 & a_13 \\
a_21 & a_22 & a_23 \\
a_31 & a_32 & a_33 \\
\end{bmatrix}
$

### High Level Process

- 1. Normalize the first column of A to q_1
- 2. Project a_2 onto q_1, orthogonalize, and then normalize
- 3. Normalize the part of a_n that is perpendicular to all a_{1}...a_{n-1} as q_n

### Step 1: Extract Columns of A

a_1 = [a_11, a_21, a_31], a_2 = ..., a_3 = ...

### Step 2: Compute First Orthogonal Vector q1

these elements q_{x} eventually make up Q, which is an orthogonal basis that is a subspace of R^n (because the columns of A span the subpsace R of dimension n as defined by the number of rows)

$q_1 = a_1/\|a_1\| $ for each index of the column a_1

or

$q_1 = [a_11, a_21, a_31]^T/\sqrt{a_{11}^2 + a_{21}^2 + a_{31}^2}  $

### Step 3: Project a_2 onto q_1

Read proj_{x}(v) as projection function proj(.) of x, onto the vector v

$proj_{q_{1}}(a_2) = (q_{1}^{T}a_{2})q_{1} $

### Step 3: Ensure orthogonality

orthogonal projection u_2 = current vector - the projection

(I think this is because if we have a triangle, and we project one dimension onto the other, then subtract that projection wrt the original vector, we get a right angle somewhere which makes it orthogonal)

Formally, subtracting the projection removes the component of a_2 that lies along q_1. This removes the part of a_2 that is parallel to q_1. **The remaining vector u_2 is the portion of a_2 that is perpendicular to q_1**

$u_{2} = a_2 - (q_{1}^{T}a_{2})q_{1}  $

### Step 4: Normalize the portion of a_2 that is perpendicular to a_1

$q_2 = u_2/\|u_2\| $ for each index of the column

### Step 5: Find the portion of a_n that is perpendicular to to all a_{1}...a_{n-1} and normalize as q_n

$u_{3} = a_{3} - (q_{1}^{T}a_{3})q_{1} - (q_{2}^{T}a_{3})q_{2}  $

$q_3 = u_3/\|u_3\| $ for each index of the column

### Finally: Q the orthogonal basis

$Q = [q_{q}, ..., q_{n}]$

### Finally: R

$R = Q^T A $

#### For Cholesky

A = LL^T = R^T R

Where L = R^T (because inverse of upper triangular is lower triangular)

### Finally: x

$Rx = Q^{T}b' = Q^{T}b  $ to solve for x

Functionally, we solve: $b' = QQ^T b  $ to see projection of b on basis

P = QQ^T is the projection matrix

### Inverse of A using QR

- $A = QR $
- $A^{-1} = (QR)^{-1} $
- $A^{-1} = R^{-1}Q^{-1} = R^{-1}Q^{T} $

### For Rectangular Matrices

Moore-Penrose pseudoinverse:

- MP = $A^{+} = R^T(RR^T)^{-1} Q^T  $

### Some final comments

- if columns of original A matrix are not linearly independent, then Gram-Schmidt will yield Q of q_1 ... q_d vectors, which are either:
  - unit-normalized vectors, or
  - zero vectors
    - these zero vectors will have zero coordinates in the Gram-Schmidt representation, since the coordinates of zero vectors are irrelevant from a representational point of view
    - here, we do regular process then drop all zero columns from Q, and drop zero rows with matching indices from R

## Schur Decomposition

- basis change matrix P is orthogonal
- upper triangular matrix U contains eigenvalues on the diagonal with no special properties
- can be found with QR decomposition
- the `Schur decomposition` of a `symmetric matrix` is the **same** as its `diagonalization`
- A `Schur decomposition` of a `real matrix` always exists
  - may not be unique
  - may be complex valued

A = P U P^T
