# Chapter 2. Linear Transformations and Linear Systems

## 2.1 Introduction

- `vector spaces`
  - the collection of row vectors and column vectors into data matrices

Mutliplying a vector can be used to implement rotation, scaling, reflection operations

A multiplication of a vector with a matrix can be shown to be some combination of rotation, scaling, and reflection being applied to the vector

### 2.1.1 What Is a Linear Transform

Typically accomplished by multiplying matrices and vectors

#### Definition 2.1.1 - Linear Transform

A vector-to-vector function f(\bar{x}) defines a linear transformation of \bar{x}, if the following conditions are satisfied for any scalar c:

$f(c\bar{x}) = c \cdot f(\bar{x}), \forall x in domain of f(.)  $

$f(\bar{x} + \bar{y}) = f(\bar{x}) + f(\bar{y}), \forall \bar{x}, \bar{y} in domain of f(.)  $

A vector-to-vector fn is a generalization of the notion of scalar functions, and it maps a d-dimensional vector to an n-dimensional vector for some d and n

the `translation` operator is NOT a linear transform

An `affine transformation` is a combination of linear transformation with a translation

- `affine transformations`
  - a combination of linear transform with a translation
  - includes any transform of the form:
    - f(x) = Ax + c
      - where both x and c are vectors
      - A is an nxd matrix
      - x is a d-dimensional vector
      - c is an n-dimensional column vector

#### Definition 2.1.2 - Affine Transform

A vector-to-vector fn f(\bar{x}) defines an affine transformation of \bar{x}, if the following condition is satisfied for any scalar \labmda

$f(\lambda \bar{x} + [1 - \lambda]\bar{y}) = \labmda f(\bar{x}) + [1 - \lambda]f(\bar{y}), \forall \bar{x}, \bar{y} in domain of f(.)  $

All linear transforms are special cases of affine transforms, but not vice versa. the simplest univariate fn f(x) = m.x + b is widely referred to as "linear", allows a non-zero translation b; this would make it an affine transformation. However, the notion of linear transform from the linear algebra perspective is much more restrictive, and it does not even include the univariate fn, unless the bias term b is zero

The class of linear transforms can always be geometrically expressed as a sequence of one or more rotations, reflections, dilations/contractions about the origin. The origin always maps to itself after these operations, and therefore translation is not included

In this book, "linear transform" or "linear operator" will be used in the context of linear algebra, where **translation is NOT allowed**

"linear function" will be used in the context of machine learning, where **translation is allowed**

## 2.2 The Geometry of Matrix Multiplication

The multiplicaiton of a d-dimensional vector with an nxd matrix is an example of a linear transformation. **The converse is also true**

#### Lemma 2.2.1 - Linear Transformation is Matrix Multiplication

Any linear mapping f(\bar{x}) from d-dimensional vectors to n-dimensional vectors can be represented as the matrix-to-vector product A\bar{x} by construcing A as follows. The columns of the nxd matrix A are $f(\bar{e}_{1}...f(\bar{e}_{d}))$, where \bar{e}_{i} is the ith column of the dxd identity matrix

Setting A to the scalar m yields a special case of the scalar-to-scalar linear fn f(x) = m.x + b (with b=0). For vector-to-vector transformations, one can either transform a row vector \bar{y} as \bar{y}V or (equivalently) transform the column vector $\bar{x} = \bar{y}^{\top} as V^{\top}\bar{x}  $:

$f(\bar{y}) = \bar{y}V  $ [Linear transform on row vector \bar{y}]

$g(\bar{x}) = V^{\top}\bar{x}  $ [Same transformation on column vector \bar{x} = \bar{y}^{\top} ]

One can also treat a matrix-to-matrix multiplication between nxd matrix D and dxd matrix V as a linear transformation of the rows of the first matrix. In other words, the ith row of the nxd matrix D' = DV is the transformed representation of the ith row of the original matrix D.

Data matrices in ML often contain multidimensional points in their rows

Matrix transformations can be broken up into geometrically interpretable sequences of transformations by expressing matrices as products of simpler matrices

#### Observation 2.2.1 - Matrix Product as Sequence of Geometric Transformations

The geometric transformation caused by multiplying a vector with V = V_{1}V_{2}...V_{r} can be viewed as a sequence of simpler geometric transformations by regrouping the product as follows:

$\bar{y}V = ([\bar{y}V_{1}V_{2}]...V_{r})\bar{y}  $ For row vector

$V^{\top}\bar{x} = (V_{r}^{\top}[V_{r-1}^{\top}...(V_{1}^{\top})])\bar_{x} = \bar{y}^{\top}  $ For column vector

#### Orthogonal Transformations

The orthogonal 2x2 matrices V_{r} and V_{c} that respectively rotate 2-dimensional row and column vectors by \theta degress in the counter clockwise counter-clockwise direction are as follows:

$ V_{r} =
\begin{bmatrix}
\cos{\theta} & \sin{\theta} \\
-\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

$ V_{c} =
\begin{bmatrix}
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

DV_{r} will rotate each **row** of D

V_{c}D^{\top} will rotate each **column** of d

The two columns of the transformation matrix V_{r} represent the mutually orthogonal unit vectors of a new axis system that is rotated clockwise by \theta

Orthogonal matrices might include reflections

Consider V = [[0, 1],[1, 0]]. Here, the transformation DV of the rows D simply flips the two coordinates in each row of D. The resulting transformation cannot be expressed purely as a rotation. This is because this transformation changes the `handedness` of the data

In some cases, reflections are included with rotations. When a compulsory reflection is included in the sequence, the resulting matrix is referred to as a `rotreflection` matrix

#### Lemma 2.2.2 - Closure Under Multiplication

The product of any number of othogonal matrices is always an orthogonal matrix

What about the commutativity of the product of orthogonal matrices? Basically, not really commutative in higher dimensions. the main issue is that rotations in higher dimensions are associated with a vector referred to as the `axis of rotation`. orthogonal matrices that do not correspond to the same `axis of rotation` may not be commutative

All 3-dimensional rotation matrices can be geometrically expressed as a single rotation, albeit with an arbitrary axis of rotation

#### Givens Rotations and Householder Reflections

It is not possible to express a rotation matrix using a single angle in dimensionalities greater than 3. In such cases, independent rotations of different angles might be occuring in unrelated planes (e.g. xy-plane and zw-plane). therefore, one must express a rotation transformation as a *sequence* of elementary rotations, each of which occurs in a 2-dimensional plane

One natural choice for defining an elementary rotation is the `Givens rotation`, which is a generalization of eq. 2.4 to higher dimension

A dxd `Givens rotation` always selects two coordinate axes and performs the rotation in that plane, so that post-multiplying a d-dimensional row vector with that rotation matrix changes only two coordinates.

The dxd Givens rotation matrix is different from the dxd identity matrix in only 2x2 relevant entries; these entries are the same as those of a 2x2 rotation matrix

A 4x4 givens rotation matrix is denoted as G_{r}(2, 4, \alpha). The notation G(.,.,.) for row-wise and column-wise transformation matrices are respectively sub-scripted by either "r" or "c". All orthogonal matrices can be decomposed into Given rotations, athougha a reflection might also be needed

#### Lemma 2.2.3 - Givens Geometric Decomposition

All dxd orthogonal matrices can be shown to be products of at most O(d^{2}) Givens rotations and at most a single elementary reflection matrix (obtained by negating one diagonal element of the identity matrix)

#### Problem 2.2.1

Show that you can express a dxd elementary row interchange matrix as the product of a 90deg rotation and an elementary reflection
