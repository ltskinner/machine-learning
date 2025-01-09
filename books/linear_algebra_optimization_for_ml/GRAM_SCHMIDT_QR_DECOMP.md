# Gram-Schmidt Process / QR Decomposition

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

## High Level Process

- 1. Normalize the first column of A to q_1
- 2. Project a_2 onto q_1, orthogonalize, and then normalize
- 3. Normalize the part of a_n that is perpendicular to all a_{1}...a_{n-1} as q_n

## Step 1: Extract Columns of A

a_1 = [a_11, a_21, a_31], a_2 = ..., a_3 = ...

## Step 2: Compute First Orthogonal Vector q1

these elements q_{x} eventually make up Q, which is an orthogonal basis that is a subspace of R^n (because the columns of A span the subpsace R of dimension n as defined by the number of rows)

$q_1 = a_1/\|a_1\| $ for each index of the column a_1

or

$q_1 = [a_11, a_21, a_31]^T/\sqrt{a_{11}^2 + a_{21}^2 + a_{31}^2}  $

## Step 3: Project a_2 onto q_1

Read proj_{x}(v) as projection function proj(.) of x, onto the vector v

$proj_{q_{1}}(a_2) = (q_{1}^{T}a_{2})q_{1} $

## Step 3: Ensure orthogonality

orthogonal projection u_2 = current vector - the projection

(I think this is because if we have a triangle, and we project one dimension onto the other, then subtract that projection wrt the original vector, we get a right angle somewhere which makes it orthogonal)

Formally, subtracting the projection removes the component of a_2 that lies along q_1. This removes the part of a_2 that is parallel to q_1. **The remaining vector u_2 is the portion of a_2 that is perpendicular to q_1**

$u_{2} = a_2 - (q_{1}^{T}a_{2})q_{1}  $

## Step 4: Normalize the portion of a_2 that is perpendicular to a_1

$q_2 = u_2/\|u_2\| $ for each index of the column

## Step 5: Find the portion of a_n that is perpendicular to to all a_{1}...a_{n-1} and normalize as q_n

$u_{3} = a_{3} - (q_{1}^{T}a_{3})q_{1} - (q_{2}^{T}a_{3})q_{2}  $

$q_3 = u_3/\|u_3\| $ for each index of the column

## Finally: Q the orthogonal basis

$Q = [q_{q}, ..., q_{n}]$

## Finally: R

$R = Q^T A $

## Finally: x

$Rx = Q^{T}b' = Q^{T}b  $ to solve for x

## Some final comments

- if columns of original A matrix are not linearly independent, then Gram-Schmidt will yield Q of q_1 ... q_d vectors, which are either:
  - unit-normalized vectors, or
  - zero vectors
    - these zero vectors will have zero coordinates in the Gram-Schmidt representation, since the coordinates of zero vectors are irrelevant from a representational point of view
    - here, we do regular process then drop all zero columns from Q, and drop zero rows with matching indices from R
