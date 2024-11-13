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

#### Definition 2.3.3 - Subspace

A vector space S is a subspace of another vector space V, if any vector x \in S is also present in V. In addition, when V contains vectors not present in S, the subspace S is a `proper subspace` of V

The requirement that subspaces are vector spaces ensures that subspaces of \mathbb{R}^{n} contain vectors residing on hyperplanes in n-dimensional space passing through the origin

When the hyperplane defining the subspace has dimensionality strictly less than n, the corresponding subspace is a proper sub-space of R^n because non-hyperplane vectors in R^n are not members of the subspace

if you have [1, 0, 0]^T and [1, 2, 1]^T, these can be used to define a 2-dimensional hyperplane V, and each point on the hyperplane is a linear combination of the pair of vectors

then, when using linear combinations to create new points, any subspace between two of these points must be a proper subspace of V

For the vector space R^3, examples of proper subspaces could be the set of vectors sitting on:

- i. any 2-dimensional plane passing through the origin
- ii. any 1-dimensional line passing through the origin
- iii. the zero vector

Furthermore, subspace relationships might exist among the lower-dimensional hyperplanes when one of them contains the other (e.g. a 1-dimensional line sitting on a plane in R^3)

A set of vectors {a_1...a_d} is `linearly dependent` if a non-zero linear combination of these vectors sums to zero

#### Definitino 2.3.4 - Linear Dependence

A set of non-zero vectors $\{\bar{a}_{1}...\bar{a}_{d} \}  $ is linearly dependent, if a set of d scalars $x_{1}...x_{d}  $ can be found so that at least some of the scalars are non-zero, and the following condition is satisfied

$\sum_{i=1}^{d} x_{i}\bar{a}_{i} = \bar{0}  $

We emphasize the fact that all scalars x_1...x_d cannot be zero. Such a coefficient set is said to be `non-trivial`

`linearly independent` when no set of non-zero scalars can be found

It is relatively easy to show that a set of vectors a...a that are mutually orthogonal must be linearly independent. If these are linearly dependent, we must have non-trivial coefficients x_1...x_d such that $\sum_{i=1}^{d} x_{i}\bar{a}_{i} = \bar{0}  $

However, taking the dot product of the linear dependence condition with each a_i and setting each a_i \cdot a_j = 0 for i \neq j yields each x_i = 0, which is a `trivial` coefficent set

Building on above example of having two vectors defining a plane, and then using linear combinations to create new points. One only needs 2 of these vectors to define the hyperplane on which all vectors lie. This **minimal set of vectors** is known as a `basis`

#### Definition 2.3.5 - Basis

A `basis` (or `basis set`) of a vector space $V \subseteq \mathbb{R}^{n} $ is a **minimal** set of vectors $\mathbb{B} = \{\bar{a}_{1}...\bar{a}_{d} \} \subseteq V  $, so that all vectors in V can be expressed as linear combinations of a_1...a_d

In other words, for any vector v \in V,, we can find scalars x_1...x_d so that $\bar{v} = \sum_{i=1}^{d} x_{i}\bar{a}_{i}  $, and one cannot do this for any proper subset of B

It is helpful to think of a basis geometrically as a coordinate system of directions or `axes`, and the scalars x_1...x_d as coordinates in order to express vectors

For example, the two commonly used axis directions in the classical 2d plane of Cartesian geometry are [1,0]^T and [0,1]^T, although we could always rotate this axis system by \theta to get a new set of axes

Furthermore, the representative directions need not even be mutually orthogonal. The basis set does not need to be unique

The vectors in a basis must be linearly independent. this is because if the vectors in the basis are linearly dependent, we can drop any vector occuring in the linear dependence condition from B without losing the ability to express all vectors in V in terms of the remaining vectors. "if any vector in a supposed basis were linearly dependent on others, it could be removed without reducing the span. This means that a dependent vector in a basis would be redundant because it doesnt contribute a new, independent direction"

Furthermore, if the linear combination of a set of vectors B cannot express a particular vector in v \in V, one can add v to the basis set without disturbing its linear independence. This process can be continued until all vectors in V are expressed by a linear combination of the set B. Therefore, an alternative definition for `basis` is:

#### Definition 2.3.6 - Basis: Alternative Definition

A basis (or basis set) of a vector space V is a **maximal** set of linearly independent vectors in it

Both definitions of the basis are equivalent and can be derived from one another. An interesting artifact is that the vector space containing only the zero vector has an empty basis

A vector space containing non-zero vectors always has an infinite number of possible basis sets

The `dimension theorem of vector spaces` states that the size of every basis set of a vector space must be the same

#### Theorem 2.3.1 - Dimension Theorem for Vector Spaces

The number of members in every possible basis set of a vector space V is always the same. This value is referred to as the **dimensionality** of the vector space

The notion of subspace dimensionality is identical to that of geometric dimensionality of hyperplanes in R^n

For example, any set of n linearly independent directions in R^n can be used to create a basis (or coordinate system) in R^n

For subspaces corresponding to lower dimensional hyperplanes, we only need as many linearly independent vectors sitting on the hyperplane as are needed to uniquely define it. This value is the same as the geometric dimensionality of the hyperplane

#### Lemma 2.3.1 - Matrix Invertibility and Linear Independence

An nxn square matrix A has linearly independent columns/rows if and only if it is invertible

When vector spaces contain abstract objects like degree-p polynomials of the form $\sum_{i=0}^{p} c_{i}t^{i}  $, the basis contains simple instantiations of these objects like $\{t^0, t^1,...t^p \}  $. Choosing a basis like this allows us to use the coefficients $[c_0...c_p]^T$ of each polynomial as the new vectors space $\mathbb{R}^{p+1} $

Carefully chosen basis sets *allow us to automatically map all d-dimensional vector spaces over real fields to R^d for finite values of d*

For example, V might be a d-dimensional subspace of R^n (for d < n). However, once we select d basis vectors, the set of d-dimensional combination coefficients for these vectors themselves create the "nicer" vector space R^d (opposed to R^n). Therefore, we have a one-to-one isomorphic mapping between any d-dimensional vector space V and R^d

### 2.3.1 Coordinates in a Basis System

Let $\bar{v} \in \subset R^n  $ by a vector drawn from a d-dimensional vector space V for d < n. In other words, the vector space contains all vectors sitting on a d-dimensional hyperplane in R^n. The coefficients in x_1...x_d, in terms of which the vector $\bar{v} = \sum_{i=1}^d x_{i} \bar{a}_i $ is represented in a particular basis are referred to as its coordinates. A particular basis set of the vector space R^n referred to as the *standard basis*, contains the n-dimensional column vectors $\{ \bar{e}_{1}, ... \bar{e}_{n} \}  $ where each e_i contains a 1 in the ith entry and a value of 0 in all other entries

The standard basis set is often chosen by default, where the scalar components of vectors are the same as their coordinates. However, scalar components of vectors are not the same as their coordinates for arbitrary basis sets. The standard basis is restrictive because it cannot be used as the basis of a `proper` subspace of R^n

The coordinates of a vector in any basis must be unique

#### Lemma 2.3.2 - Uniqueness of Coordinates

The coordinates $\bar{x} = [x_1 , ..., x_d]^T  $ of any vector $\bar{v} \in V $ in terms of a basis set $B = \{\bar{a}_{1}...\bar{a}_{d} \}  $ are always unique

How can one find these unique coordinates? When $\bar{a}_{1}...\bar{a}_{d} $ correspond to an **orthogonal** basis of V, the coordinates are simply the dot products of \bar{v} with those vectors

By taking the dot product of both sides of $\bar{v} = \sum_{i=1}^{d} x_{i}\bar{a}_{i}  $ with each $\bar{a}_{j}$ and using orthonormality, it is easy to show that $x_j = \bar{v} \cdot \bar{a}_j$

For non-orthogonal basis systems, it is much harder to find the coordinates. The general problem is that of:

solving the system of equations $A\bar{x} = \bar{v} $ for $\bar{x} = [x_1 ... x_d]^T  $

where the n-dimensional columns of the nxd matrix A contain the (linearly independent) basis vectors. The problem boils down to finding a solution to the system of equations $A\bar{x} = \bar{v} $ where $A = [\bar{a}_{1}...\bar{a}_{d}]  $ contains the basis vectors ofo the d-dimensional vector space $V \subseteq R^n  $. Note that the basis vectors are themselves represented using n components like the vectors of R^n, even though the vector space V is a d-dimensional subspace of R^n and the coordinate vector \bar{x} lies in R^d. If d=n, and the matrix A is square, the solution is simply $\bar{x} = A^{-1}\bar{v} $

However, when A is not square, one may not be able to find valid coordinates, if \bar{v} does not lie in $V \subset R^n  $. This occurs when \bar{v} does not geometrically lie on the hyperplane H_A defined by all possible linear combinations of the columns of A.

However, one can find the *best fit* coordinates \bar{x} by observing that:

- the line joining the closest linear combination $A\bar{x}  $ of the columns of A to \bar{v} must be **orthogonal** to the hyperplane H_A, and is therefore orthogonal to every column of A. The condition that $(A\bar{x} - \bar{v})$ is orthogonal to every column of A can be expressed as the `normal equation` $A^T (A\bar{x} - \bar{v}) = \bar{0} $ which results in:

$\bar{x} = (A^T A)^{-1} A^T  \bar{v} $

The best-fit solution includes the exact solution when it is possible

- `left-inverse` of matrix A
  - $\bar{x} = (A^T A)^{-1} A^T  \bar{v} $
  - with linearly dependent columns

![alt-text](./2_3_basis.PNG)

Figure 2.5 - THIS IS NOT MY DIAGRAM ALL CREDIT TO THE AUTHOR

Although the notion of a non-orthogonal coordinate system does exist in analytical geometry, it is rarely used in practice because of loss of visual interpretability of coordinates. However, such non-orthogonal basis systems are very natural to linear algebra, where some loss of geometric intuition is often compensated by algebraic simplicity

### 2.3.2 Coordinate Transformations Between Basis Sets

Previous section talks about how different basis sets correspond to different coordinate systems

What if we have coordinates $\bar{x}_{a}$ defined wrt to the basis set $\{a_{1}, ..., \bar{a}_{n} \}  $ of R^n into the coordinates $\bar{x}_{b}$ defined wrt to the n-dimensional basis set $\{\bar{b}_{1}, ..., \bar{b}_{n} \} $. The goal is to find an nxn matrix $P_{a\rightarrow b}  $ that transforms $\bar{x}_{a} to \bar{x}_{b}  $:

$\bar{x}_b = P_{a \rightarrow b}\bar{x}_{a}  $

For ex: how might one transform the coordinates in the orthogonal basis set of figure 2.5.b into the non-orthogonal system of 2.5.c ?

Here, the key point is to observe that the coordinates x_a and x_b are **representations of the same vector**, and they would therefore have the **same coordinates** in the standard basis.

First, we use the basis sets to construct two nxn matrices: $A = [\bar{a}_{1}...\bar{a}_{n}]  $ and $B = [\bar{b}_{1}...\bar{b}_{n}]  $ since the oordinates \bar{x} of $\bar{x}_{a}  $ and $\bar{x}_{b} $ must be identical in the standard basis, so we have:

$A\bar{x}_a = B\bar{x}_{b} = \bar{x}  $

Already established that square matrices defined by linearly independent vectors are invertible - therefore mulitplying both sides with B^{-1} we obtain:

$\bar{x}_b = [B^{-1}A]\bar{x}_{a}  $ where $[B^{-1}A]$ is $P_{a\rightarrow b}  $

The main computational work involved in the transformation is in inverting the matrix B

One observation is that when B is an orthogonal matrix, the transformation matrix simplifies to B^{T}A

And, when A corresponds to the standard basis, the transformation matrix is B^T (because identity)

Therefore, working with orthonormal bases simplifies computations, which is why the identification of orthonormal basis sets is an important problem in its own right

---

It is also possible to perform coordinate transformations between basis sets that define a particular d-dimensional *subspace* V of R^n, rather than all of R^n.

Let $\{\bar{a}_1...\bar{a}_d \} and \{\bar{b}_1...\bar{b}_d \} $ be two basis sets for this d-dimensional subspace V, such that each of these basis vectors is expressed in terms of the standard basis of R^n.

Furthermore, let $\bar{x}_{a} and \bar{x}_{b}  $ be two d-dimensional coordinates of the same vector \bar{v} \in V in terms of the two basis sets. We want to transform the known coordinates of x_a to the unknown coordinates of x_b in the second basis set (and find a best fit if the two basis sets represent different vector spaces). In this case, because matrix B is not square, it cannot be inverted to solve for x_b in terms of x_a, and we sometimes might have to be content with a best fit

Here, we use the normal equation and $A\bar{x}_a - B\bar{x}_b  $ needs to be orthogonal to every column of B in order to be a best-fit solution. This implies that:

$B^T (A \bar{x}_{a} - B\bar{x}_{b}) = \bar{0}  $

and we have the following:

$\bar{x}_b = (B^T B)^{-1} B^T A \bar{x}_{a}  $ where $(B^T B)^{-1} B^T A $ is $P_{a\rightarrow b}  $

when B is square and invertible, it is easy to show that this solution simplifies to $B^{-1}A \bar{x}_{a}  $

### 2.3.3 Span of a Set of Vectors

Even though a vector space is naturally defined by a basis set (which is linearly independent), one can also define a vector space by using a set of linearly dependent vectors. This achieves the notion of a `span`

#### Definition 2.3.7 - Span

The span of a finite set of vectors $A = \{\bar{a}_1, ..., \bar{a}_{d}  \}  $ is the vector space defined by all possible linear combinations of the vectors in A:

$Span(A) = \{\bar{v}: \bar{v} = \sum_{i=1}^{d} x_{i}\bar{a}_{i}, \forall x_1...x_d \in \mathbb{R}  \}  $

Points that do not lie on the hyperplane defined by two vectors are not in the span. New vectors that are linearly dependent on the basis vectors are naturally in the span

When the set A contains linearly independent vectors, it is also a basis set of its span

If have three vectors that all lie on a hyperplane passing through the origin, their span is all the vectors on the hyperplane

If have three linearly independent vectors, the span is all vectors in R^3. For the linearly indepdnent vectors spanning R^3, these can be used to create a valid coordinate system to represent any vector in R^3 (albeit a non-orthogonal one)

### 2.3.4 Machine Learning Example: Discrete Wavelet Transform

Basis transformations are used frequently in ML of time series

vectors length of n, where each n is a time step, so an hour would be R^3600

a common charactersitic of time series is that consequtive values are very similar in most real applications. therefore, most information would be hidden in a few variations across time. The Haar wavelet transformation performs precisely a basis transformation that extracts the important variations. Typically, only a few such differences will be large, which results in a sparse vector. Aside from the space-efficiency advantages of doings so, some predictive algorithms seem to work better with coordinates that reflect tred differences

consider $\bar{s} = [8, 6, 2, 3, 4, 5, 6, 6, 5]^T in R^8  $ - the representaiton corresponds to the values in the standard basis. however, we want a basis in which the differnces between *continguous* regions of the series are emphaized. Therefore, we define the following set of 8 vectors to create a new basis in R^8 together with an interpretation of what their coefficients represent wo within a proportionality factor

note that all basis vectors are othogonal, though they are not notamized to unit norm. we want to transform the time series from the std basis into the new set of orthogonal vectors
