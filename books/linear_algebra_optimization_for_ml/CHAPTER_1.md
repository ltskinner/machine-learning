# Chapter 1. Linear Algebra and Optimization: An Introduction

## 1.1 Introduction

ML builds mathematical models from data containing multiple *attributes* (i.e. variables) in order to predict some variables from others. Such models are expressed as linear and nonlinear relationships between variables. These relationships are discovered in a data-driven manner by optimizing (maximizing) the "agreement" between the models and the observed data - this is an optimization problem.

Linear algebra is the study of linear operations in *vector spaces*. ex. the infinite set of all possible cartesian coordinates in two dimensions in relation to the origin

Dimensions correspond to attributes in ML

Linear algebra can be viewed as a generalized form of the geometry of cartesian coordinates in d dimensions. Just as one can use analyticsl geometry in two dimensions in order to find the intersection of two lines on a plane, one can generalize this concept to any number of dimensions. The resulting method is referred to as *Gaussian elimination* for solving systems of equations. The probelm of *linear regression*, which is fundamental to linear algebra, optimization, and machine learning is closely related to solving systems of equations

## 1.2 Scalars, Vectors, Matrices

1. Scalars
2. Vectors
3. Matrices - always upper case variables

Pythagorean theorum - a^2 + b^2 = c^2

in linear algebra, only vectors who have tails at the origin are considered - **all vectors, operations, and spaces in linear algebra use the origina as an important referecne point**

### 1.2.1 Basic Operations with Scalars and Vectors

Vectors of the same dimensionality can be added or subtracted

Vector addition and subtraction are commutative - the order of numbers does not matter

- $\bar{x} + bar{y} = [x_{1}...x_{d}] + [y_{1}...y_{d}] =  [x_{1} + y_{1} ... x_{d}+y_{d}]  $
- $\bar{x} - \bar{y} = [x_{1}...x_{d}] - [y_{1}...y_{d}] =  [x_{1} - y_{1} ... x_{d}-y_{d}]  $

Scalar Multiplication

- $\bar{x}' = a \bar{x} = [ax_{1}...ax_{d}]  $

This operation scales the length of the vector, but does not change its direction

Dot product multiplication

- $\bar{x} \cdot \bar{y} = \sum_{i=1}^{d} x_{i}y_{i}  $

[1, 2, 3] \cdot [6, 5, 4] = (1)(6) + (2)(5) + (3)(4) = 28

The dot product is a special case of a more general operation, referred to as the *inner product*, and it preserves many fundamental rules of Euclidian geometry. The space of vectors that includes a dot product operation is referred to as the Euclidian space. It is also commutative.

It is also distributive

- $\bar{x} \cdot (\bar{y} + \bar{z}) = \bar{x} \cdot \bar{y} + \bar{x} \cdot \bar{z}  $

The **dot product of a vector with itself** is referred to as its squared norm, or Euclidian norm. The norm defines the vector length and is denoted by $\|.\|$

- $\|\bar{x} \|^{2} = \bar{x} \cdot \bar{x} = \sum_{i=1}^{2} x_{i}^{2} $

The norm of the vector is the Euclidian distance of its coordinates from the origin

Often vectors are **noramlized** to unit length by dividing them with their norm - unit vector

- $\bar{x}' = \frac{\bar{x}}{\|\bar{x}\|} = \frac{\bar{x}}{\|\sqrt{\bar{x}\cdot\bar{x}}}  $ 

Scaling a vector by its norm does not change the relative values of its components

A generalization of the Euclidian norm is the L_{p}-norm, which is denoted by $\|.\|_{p} $

$\|\bar{x}\|_{p} = (\sum_{i=1}^{d} |x_{i}^{p}|  )^{1/p}  $

Here, |.| indicates the absolute value of a scalar, and p is a positive integer. When p is set to 1, the resulting norm is referred to as the Manhattan norm or L_{1}-norm

The (squared) Euclidian distance between \bar{x} and \bar{y} can be shown to be the doct product of \bar{x} - \bar{y} with itself:

$\|\bar{x} - \bar{y}\|^{2} = (\bar{x} - \bar{y}) \cdot (\bar{x}-\bar{y} ) = sum_{i=1}^{d}(x_{i} - y_{i})^{2} = Euclidean(\bar{x}, \bar{y})^{2}  $

Dot products satisfy the *Cauchy-Schwarz inqeuality*, accoding to which the dot product between a pair of vectors id bounded above by the product of their lengths

$|\sum_{i=1}^{d}x_{i}y_{i}| = |\bar{x} \cdot \bar{y}  | \leq  \|\bar{x}\| \|\bar{y}\|  $

The ratio between the two quantities is the cosine of the angle between the two vectors

- `polar form`
  - $[a, \theta]$
  - a is the length of the vector
  - \theta is the counter clock wise angle the vector makes with the X axis
- `cartesian form`
  - $[a \cos(\theta), a \sin(\theta)]  $
  - the dot porduct with [0,1] (the X-axis) is $a \cos(\theta)  $

The cosine fn between two vectors is algebraically defined by the dot product between the two vectors *after* scaling them to unit norm:

$\cos(\bar{x}, \bar{y}) = \frac{\bar{x} \cdot \bar{y}}{\sqrt{\bar{x}\cdot \bar{x}} \sqrt{\bar{y}\cdot \bar{y}}} = \frac{\bar{x} \cdot \bar{y}}{\|\bar{x}\| \|\bar{y}\|}  $

`cosine law` from Euclidean geometry:

$\cos(\theta) = \frac{a^{2} + b^{2} - c^{2}}{2ab} = \frac{\|\bar{x}\|^{2} + \|\bar{y}\|^{2} - \| \bar{x} - \bar{y}\|^{2}}{2(\|\bar{x}\|)(\|\bar{y}\|)} = \frac{\bar{x}\bar{y}}{\sqrt{\bar{x}\cdot \bar{x}}\sqrt{\bar{y}\cdot\bar{y}}}  $

A pair of vectors is `orthogonal` if their dot product is 0 (and the angle between them is 90). The vector \bar{0} is considered orthogonal to every vector.

A set of vectors is `orthonormal` if each pair in the set is mutually orthogonal and the norm of each vector is 1.

Orthonormal directions are useful because they are employed for transformations of points across different orthogonal coordinate systems with the use of 1-dimensional `projections` 

- `coordinate transformation` or `projection`
  - compute a new set of coordinates with respect to a changed set of directions

Consider the point [10, 15]. Imagine given two orthonormal directions [3/5, 4/5] and [-4/5, 3/5] - compute the dot product to project

- x' = 10*(3/5) + 15*(4/5) = 18
- y' = 10*(-4/5) + 15*(3/5) = 1

These types of transformations of vectors to new representaitons lie at the heart of linear algebra. In many cases, transformed representations of data sets have useful properties, which are exploited by ML applications.

### 1.2.2 Basic Operations with Vectors and Matrices

The `transpose` of a matrix is obtained by flipping its rows and columns. The (i,j)th entry of the transpose is the same as the (j,i)th of the original. comes from an n x d matrix to a d x n matrix, denoted as $A^{\top} $

Like vectors, matrices can be added only if they have the same size

- 3x2 times 2x1 (inner dimensions the same)
  - --> 3x1 (outer dimensions become dimensions of resultant)
- 1x3 x 3x2 = 1x2

The multiplication of an nxd matric with a d dimensional column vector creates an n-dimensional column vector, is often interpreted as a *linear transformation* from d-dimensional space to n-dimensional space

the nxd matrix A is occasionally represented in terms of its ordered set of n-dimensional columns $\bar{a}_{1}...\bar{a}_{d} $ as $A [\bar{a}_{1}...\bar{a}_{d}]  $. This results in the following form of *matrix-vector multiplication* using the columns of A and a column vector $\bar{x} = [x_{1}...x_{d}]^{\top}  $ of coefficients:

$A\bar{x} = \sum_{i=1}^{d}x_{i}\bar{a}_{i} = \bar{b}  $

Each x_{i} corresponds to the "weight" of the i-th direction of $\bar{a}_{i}$, which is also referred to as the ith *coordinate* of \bar{b} using the (possibly non-orthogonal) directions contained in the columns of A. This notation is a generalization of the (orthogonal) Cartesian coordinates defined by d-dimensional vectors $\bar{e}_{i}...\bar{e}_{i}  $, where each e is an axis direciton with a single 1 in the ith position and remaining 0s

The dot product between two vectors can be viewed as a special case of matrix-vector multiplication

the outer product between two vectors is a nxn matrix, denoted by $\bar{x} \cross \bar{v}  $. The "tall" matrix is alwasy ordered before the "wide" matrix:

- 3x1 times 1x3 = 3x3

Unlike dot products, the outer product can be performed between two vectors of different lengths.

Conventionally, outer products are defined between two column vectors, and the second vector is transposed into a matrix containing a single row before matmul

$(UV)_{ij} = \sum_{r=1}^{k} u_{ir}v_{rj}  $

### Problem 1.2.2 (Outer Product Properties)

Show that if an nx1 matrix is multipled with a 1xd matrix (which is also an outer product between two vectors), we obtain an nxd matrix with the following properties:

- Every row is a multiple of every other row
- Every column is a multiple of every other column

Each entry in a matrix product is an `inner product` of two vectors extracted from the matrix. What about outer products? It can be shown that the entire matrix is the sum of as many outer products as the common dimension k of the two multiplied matrices

### Lemma 1.2.1 (Matrix Multiplication as Sum of Outer Products)

The product of an nxk matrix U with a kxd matrix V results in an nxd matrix, which can be expressed as the sum of k outer-product matrices; each of these k matrices is the product of an nx1 matrix with a 1xd matrix. Each nx1 matrix corresponds to the ith column U_{i} of U, and each 1xd matrix corresponds to the ith row V_{i} of V. Therefore, we have the following:

$UV = \sum_{r=1}^{d}U_{r}V_{r} $ where the product dimensions of U_{r}V_{r} is nxd

Matrix multiplication is not *commutative*, but it is *associative* and *distributive*

$[A(BC)]_{ij} = [(AB)C]_{ij} = \sum_{k}\sum_{m} a_{ik}b_{km}c_{mj}  $

### Problem 1.2.3

Express the matrix ABC as the weightes sum of outer products of vectors extracted from A and C. The weights are extracted from matrix B

### Problem 1.2.4

Let A be an 1000000x2 matrix. Suppose you have to compute the 2x100000 matrix A^T AA^T on a computer. would you prefer to compute (A^T A)A^T or A^T (AA^T)

### 1.2.5

Let D be an nxd matrix for which eahc column sums to 0. Let A be an arbitrary dxd matrix. Show that the sum of each column of DA is also zero

The transpose of the product of two matrices is given by the prodcut of their transposes, but the order of multiplication is reversed:

$(AB)^{\top} = B^{\top}A^{\top}  $

### Problem 1.2.6

Show the following result for matrices A_{1}...A_{n}

- $(A_{1}A_{2}A_{3}...A_{n})^{\top} = A_{n}^{\top}A_{n-1}^{\top}...A_{2}^{\top}A_{1}^{\top}  $

### 1.2.3 Special Classes of Matrices

- `symmetric matrix`
  - is a square matrix that is its own transpose
  - A = A^{\top}

### Problem 1.2.7

If A and B are symmetric matrices, then show that AB is symmetric if and only if AB = BA

The `diagonal` of a matrix is defined as the set of entries for which the row and column indices are the same. Generally used for square matrices, but can be used for rectangular matrices

#### Definition 1.2.1 - Rectangular Diagonal Matrix

A rectangular diagonal matrix is an nxd matrix in which each entry (i,j) has a non-zero value if and only if i = j. Therefore, the diagonal of non-zero entries starts at the upper-left corner of the matrix, although it might not meet the lower-right corner

`block diagonal matrix` contains square blocks B_{1}...B_{r} of (possibly non-zero) entries along the diagonal. All other entries are zero. Although each block is square, they need not be of the same size.

- `triangular matrix`
  - a generalization of the diagonal matrix

#### Definition 1.2.2 - Upper and Lower Triangular Matrix

A square matrix is an `upper triangular matrix` if all entries (i,j) below its main diagonal (i.e., satisfying i > j) are zeros. A matrix is `lower triangular` if all entries (i,j) above its main diagonal (i.e., satisfying i<j) are zeros

#### Definition 1.2.3 - Strictly Triangular Matrix

A matrix is said to be *strictly* triangular if its triangular *and* all its diagonal elements are zeros. Basically, all the diagonal elements are zero, but above or below the diagonal has values

#### Lemma 1.2.2 - Sum or Product of Upper-Triangular Matrices

The sum of upper-triangular matrices is upper triangular. the product of upper-triangular matrices is upper triangular

### 1.2.4 Matrix Powers, Polynomials, and the Inverse

Square matrices can be multiplied with themselves without violating the size constraints of matrix multiplication. Multiplying a square matrix with itself many times is analogous to raising a scalar to a particular power. The nth power of a matrix is defined as:

$A^{n} = AA...A (n-times)  $

The zero-th power of a matrix is defined to be the identity matrix of the same size

`nilpotent` - when a matrix satisfies $A^{k} = 0 $ for some integer k

All polynomials of the same matrix A will commute with respect to the multiplication operator

#### Observation 1.2.1 - Commutativity of Matrix Polynomials

Two polynomials f(A) and g(A) of the same matrix A will always commute:

$f(A)g(A) = g(A)f(A)  $

- `inverse`
  - of a square matrix A is another square matrix A^{-1}
  - not all matrices have an inverse
  - $AA^{-1} = A^{-1}A = I  $

Matrices that are invertible always have the property that a non-zero linear combination of the rows does not sum to zero. Each vector direction in the rows of an invertible matrix must contribute new, non-redundant "information" that cannot be conveyed using sums, multiples, or linear combinations of other directions

When the inverse of matrix A does exist, it is unique. Furthermore, the product of a matrix with its inverse is always commutative and leads to the identity matrix

#### Lemma 1.2.3 - Commutativity of Multiplicaiton with Inverse

If the product AB of d x d matrices A and B is the identity matrix I, then BA must also be equal to I

#### 1.2.4

When the inverse of a matrix exists, it is always unique. In other words, if B_{1} and B_{2} satisfy AB_{1} = AB_{2} = I, we must have B_{1} = B_{2}

All diagonal entries of a diagonal matrix need to be non-zero for it to be invertible or have negative powers. The polynomials and inverses of triangular matrices are also triangular matrices of the same type (i.e., lower or upper triangular). A similar result holds for block diagonal matrices.

### Problem 1.2.8 - Inverse of Triangular Matrix is Triangular

Basically, front load an inverse of R to isolate x on one side. then the k-th index of column vector e you just multiply by the corresponding row of R^-1. The inverse of R must be upper-triangular cause i think thats just a theorum or law or whatever. idk how to prove it but im pretty sure thats the deal

### Problem 1.2.9 - Block Diagonal Polynomial and Inverse

Suppose that you have a block diagonal matrix B, which has blocks B_{1}...B_{r} along the diagonal. Show how you can express the polynomial function f(B) and the inverse of B in terms of functions on block matrices

$f(B) = a_{0}I + a_{1}B^{1} + a_{2}B^{2} + ... + a_{n}B^{n}  $

$f(B) =  $
$
\begin{bmatrix} 
f(B_{1}) & 0 \\ 
0 & f(B_{2}) 
\end{bmatrix}
$

#### Problem 1.2. 10

Suppose that the matrix B is the inverse of matrix A. Show that for any positive integer n, the matrix B^{n} is the inverse of matrix A^{n}

The inversion and the transposition operations can be applied in any order without affecting the result:

$(A^{\top})^{-1} = (A^{-1})^{\top}  $

- `orthogonal matrix`
  - square matrix whose inverse is its transpose
  - $AA^{\top} = A^{\top}A = I  $

Although such matrices are formally defined in terms of having orthonomal *columns*, the commutativity in the above relationship implies the remarkable property that they contain both orthonormal columns and orthonormal rows

#### Dude, closing the loop

A useful property of invertible matrices is that they define a uniquely solvable system of equations. For example, the solution to $A\bar{x} = \bar{b}  $ exists and is uniquely defined as $\bar{x} = A^{-1}\bar{b}  $ when A is invertible

One can also view the solution $\bar{x}  $ as a new set of coordinates of $\bar{x} $ in a different (and possibly non-orthogonal) coordinate system defined by the vectors contained in the columns of A.

Note that when A is orthogonal, the solution simplifies to $\bar{x} = A^{\top}\bar{b}  $, which is equivalent to the dot product between $\bar{b} $ and each column of A to compute the corresponding coordinate.

In other words, we are **projecting** $\bar{b} $ on each orthonormal column of A to compute the corresponding coordinate

### 1.2.5 The Matrix Inversion Lemma: Inverting the Sum of Matrices

The *limit* of $A^{n}$ as $n \implies \inf  $ is the zero matrix (0). For such matrices, the following result holds:

$(I+A)^{-1} = I - A + A^{2} - A^{3} + A^{4} + ... + Infinite Terms  $

$(I-A)^{-1} = I + A + A^{2} + A^{3} + A^{4} + ... + Infinite Terms  $

This result can be used for inverting triangular matrices (although more straightforward alternative exist)

#### Problem 1.2.11 - Inverting Triangular Matrices

A dxd triangular matrix L with non-zero diagonal entries can be expressed in the for (\Delta + A), where \Delta is an invertible diagonal matrix and A is a *strictly* triangular matrix. Show how to compute the inverse of L using only diagonal matrix inversions and matrix multiplications/additions. Note that the strictly triangular matrices of size dxd are always nilpotent and satisfy A^{d} = 0

$ L = (\Delta + A), and \Delta \Delta^{-1} = I $

so

$L^{-1} = (\Delta + A)^{-1} = (D(I + D^{-1}A))^-1  $

apply *inverse rule for products*

$L^{-1} = (I + D^{-1}A)^{-1}D^{-1}  $

and now can apply the series expansion

$L^{-1} = I - D^{-1}A + (D^{-1}A)^{2} - (D^{-1}A)^3 + ...  $

remember, (AB)^3 is the same as (AB)(AB)(AB), so in this case:

$(D^{-1}A)^3 = (D^{-1}A)(D^{-1}A)(D^{-1}A) $
