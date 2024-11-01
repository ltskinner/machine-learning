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
