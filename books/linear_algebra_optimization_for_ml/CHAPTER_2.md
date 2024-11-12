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

$f(\lambda \bar{x} + [1 - \lambda]\bar{y}) = \lambda f(\bar{x}) + [1 - \lambda]f(\bar{y}), \forall \bar{x}, \bar{y} in domain of f(.)  $

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

$V^{\top}\bar{x} = (V_{r}^{\top}[V_{r-1}^{\top}...(V_{1}^{\top})])\bar{x} = \bar{y}^{\top}  $ For column vector

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

Thus far, introduced only diagonal reflection matrices that flip the sign of a vector component.

The `Householder reflection matrix` is an orthogonal matrix that reflects a vector \bar{x} into any "mirror" hyperplane of arbitrary orientation. Such a hyperplane passes through the origin and its orientation is defined by an `arbitrary` normal vector \bar{v} (of unit length)

The distance of \bar{x} from the "mirror" hyperplane is $c = \bar{x} \cdot \bar{v}$. An object and its mirror image are separated by twice this distance (c) along \bar{v}. Therefore, to perform the reflection of \bar{x} and create its mirror image \bar{x}', one must subtract *twice* of cv from x

$(I - 2\bar{v}\bar{v}^{\top}) \bar{x}  $ Householder

For any unit (column) vector \bar{v}, the matrix $(I - 2\bar{v}\bar{v}^{\top})$ is an elementary reflection matrix in the hyperplane perpendicular to \bar{v} and passing through the origin.

Any orthogonal matrix can be represented with fewer Householder reflections than Given rotations, therefore, the Householder is a more expressive transformation

#### Lemma 2.2.4 - Householder Geometric Decomposition

Any orthogonal matrix of size dxd can be expressed as the product of at most d Householder reflection matrices

#### Problem 2.2.2 - Reflection of a Reflection

Verify algebraically that the square of the householder reflection matrix is the I matrix

two unit vectors produce a matrix that is Idempotent

#### Problem 2.2.3

Show that the elementary reflection matrix, which varies from the id matrix only in terms of flipping the sign of the ith diagonal element, is a special case of the householder reflection matrix

#### Problem 2.2.4 - Generalized Householder

Show that a sequence of k mutually orthogonal Householder transformations can be expressed as $I - 2QQ^{\top} $ for a dxk matrix Q containing orthonormal columns. Which (d-k)-dimensional plane is that reflection in

#### Rigidity of Orthogonal Transformations

Dot products and Euclidean distances between vectors are unaffected by multiplicative transformations with orthogonal matrices. This is because an orthogonal transformation is a sequence of rotations and reflections, which **does not change lengths and angles**

This also means that orthogonal transformations preserve the:

- sum of squares of Euclidean distances of the data points (i.e. rows of a data matrix D) about the origin
  - this is also the (squared) Frobenius norm/energy of the nxd matrix D

When the nxd matrix D is multiplied with the dxd orthogonal matrix V, can express the Frobenius norm of DV in terms of the trace operator:

$\|DV\|_{F}^{2} = tr[(DV)(DV)^{\top}] = tr[D(VV^{\top})D^{\top}] = tr(DD^{\top}) = \|D\|_{F}^{2}   $

- `rigid` transformations
  - transformations that preserve distances between pairs of points
  - rotations and reflections not only preserve distances between points, but also absolute distances of points from the origin

Translations (which are not linear transforms) are also not rigid because they preserve distances between pairs of transformed points. However, translations usually do not preserve distances from the origin

#### Scaling: A Non-rigid Transformation

In general, a multiplication of vector x with an arbitrary matrix V might change it length. If such a matrix can be decomposed into simpler geometric operator matrices as $V = V_{1}V_{2}...V_{r} $, it means that there must be some fundamental geometric transformation V_{i} among these operator matrices that does not preserve distances.

This fundamental transformation is that of `dilation/contraction` (or more generally, `scaling`)

It is possible for these values to be negative, which also makes it a reflection operation

#### General Case: Combining Orthogonal and Scaling Transformations

Multiplying an nxd data matrix D with a diagonal matrix \Delta to create D\Delta results in scaling of the ith dimension (column) of the data matrix D with the ith diagonal entry of \Delta. This is an example of axis-parallel scaling, where the directions of scaling are aligned with the axes of representation. Just as axis-parallel scalings are performed with diagonal matrices, scalings along arbitrary directions are performed with `diagonalizable matrices`

Say we want to scale each 2-dimensional row of an nx2 data matrix

- in the direction [cos(-30), sin(-30)]
  - by a factor of 2
- in the direction [cos(60), sin(60)]
  - by a factor of 0.5

Steps:

- i. first rotate the data set D by a 30 degree angle
  - by multiplying D with orthogonal matrix V to create DV
  - DV
- ii. then multiply DV with the diagonal matrix \Delta
  - which has diagonal entries of 2 and .5
- iii. finally, rotate the dataset in the reverse direction (i.e. -30)
  - $(DV\Delta)V^{\top}  $

Transformations of the form $V\Delta V^{\top}  $ are discussed in Ch3

V = 
$
\begin{bmatrix}
\cos(-30) & \cos(60) \\
\sin(-30) & \sin(60)
\end{bmatrix}
$

=

$
\begin{bmatrix}
\cos(30) & \sin(30) \\
-\sin(30) & \cos(30)
\end{bmatrix}
$

\Delta = 

$
\begin{bmatrix}
2 & 0 \\
0 & 0.5
\end{bmatrix}
$

Not all transformations can be expressed in the form $V\Delta V^{\top}  $. However, a beautiful result, referred to as `singular value decomposition`, states that any square matrix A can be expressed in the form $A = U\Delta V^{\top}  $, where U and V are both orthogonal matrices (which might be different) and $\Delta$ is a *nonnegative* scaling matrix

Therefore, *all linear transformations defined by matrix multiplication can be expressed as a sequence of rotations/reflections, together with a single ansiotripic scaling*

## 2.3 Vector Spaces and Their Geometry

A `vector space` is an infinite `set` of vectors satisfying certain types of `set closure` properties under addition and scaling operations

One of the most important vector spaces in linear algebra is the set of all n-dimensional vectors

#### Definition 2.3.1 - Space of n-Dimensional Vectors

The space $\mathbb{R}^{n} $ consists of the set of all column vectors with n real componments

By convention, the vectors in \mathbb{R}^{n} are assumed to

- be column vectors
- have tails at the origin
- contains an infinite set of vectors
- we can scale any vector or add two vectors from R and still stay in R

#### Definition 2.3.2 Vector Space in $\mathbb{R}^{n} $

A subset of vectors V from $\mathbb{R}^{n}  $ is a vector space if it satisfies the following properties:

1. if $\bar{x} \in V  $ then $c\bar{x} \in V $ for any scalar $c \in \mathbb{R}  $
2. if $\bar{x}\bar{y} \in V  $, then $\bar{x} + \bar{y} \in V  $

The zero vector $\bar{0}  $ is included in all vector spaces becauase it always satisfies the additive identity

In general, vector spaces that are subsets of $\mathbb{R}  $ correspond to vectors sititng on an **origin-centered hyperplane** of dimensionality at most n. Therefore, vector spaces in R can be nicely mapped to our geometric understanding of lower-dimensional hyperplanes.

The **origina-centered nature of the hyperplanes is important**;

(the set of vectors with tails at the origin and heads on a hyperplane that is NOT origin centered does not define a vector space, because this set of vectors is not closed under scaling and addition)

Other than the zero vector space, all vector spaces contain an infinite set of vectors

A fixed linear transformation of each element of a vector space results in another vector space, because of the way in which linear transformations preserve the properties of addition and scalar multiplication.

The modern notation of a vector space is more general than vectors from R^n, because it allows all kinds of abstract objects to be considered "vectors" and infinite sets of such objects to be considered vector spaces (along with appropriately defined vector addition and scalar multiplication operations on these objects)

A large class of vector spaces over the Real field can be *indirectly* represented using R^n, via the process of `coordinate representation`. Furthermore, staying in R^{n} has the distinct advantage of being able to work with easily understandable operations over matrices and vectors

#### Problem 2.3.1

Let \bar{x} \in R^{d} be a vector and A be an nxd matrix. Is each of the following a vector space?

- All \bar{x} satisfying A\bar{x} = \bar{0}
  - the zero vector in all vector spaces
- All \bar{x} satisfying A\bar{x} >= \bar{0}
  - i think not. I dont like the >= operator in this context
- All \bar{x} satisfying A\bar{x} = \bar{b} for some non-zero \bar{x} \in \mathbb{R}^{n}
  - if b could assume the zero vector then it would be a vector space
- All nxn matrices in which the row sums and column sums are the same for a particular matrix (but not necessarily across matrices)
  - yeah I think so
  - a scalar times the row sums must be in the domain
  - addition an dmultiplication also holds
