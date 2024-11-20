# Chapter 3. Eigenvectors and Diagonalizable Matrices

## 3.1 Introduction

Any squaer matrix A of size dxd can be considered a linear operator, which maps the d-dimensional column vector x to the d-dimensional vector Ax. A linear transformation Ax is a combination of operations such as rotations, reflections, and scalings of a vector x

A diagonalizable matrix is a special type of linear operator. Only corresponds to a simultaneous scaling along d different directions

- d different directions are `eigenvectors`
- d scale factors are `eigenvalues`

All such matrices can be decomposed using an invertible dxd matrix V and a diagonal dxd matrix \Delta:

$A = V\Delta V^{-1}  $

The columns of V contain d eigenvectors

The diagonal entries of \Delta contain the eigenvalues

For any $\bar{x} \in R^d  $, one can geometrically interpret Ax using the decomopsition in terms of a sequence of 3 transformations:

- i. Multiplication of x with V^{-1} computes the coordinates of x in a (possibly non-orthogonal) basis system corresponsing to the columns (eigenvectors) of V
- ii. multiplication of V^{-1}x with \Delta to create $\Delta V^{-1}\bar{x} $ dilates these coordinates with scale factors in \Delta in the eigenvector directions
- iii. final multiplication with V to create $V\Delta V^{-1}\bar{x}  $ transforms the coordinates back to the original basis system (i.e. the standard basis)

The overall result is an `anisotropic scaling` in d eigenvector directions

Linear transformations that can be represented in this way correspond to diagonalizable matrices

A dxd A diagonalizable matrix represents a linear transformation corresponding to anisotropic scaling in d A linearly independent directions

When th columns of matrix V are orthonormal, $V^{-1} = V^{T}$. In this case, the scaling is done along mutually orthogonal directions, and the matrix A is always symmetric, b/c $A^{T} = V\Delta^{T}V^{T} = V \Delta V^{T} = A  $

- contraction = scale factor 0.5
- dilation = scale factor 1

## 3.2 Determinants

Imagine an object, whose outline is projected on a graph, and the shape of the outline consists of coordinate vectors

Multiplying the vectors times a square matrix will distort the object

When matrix A is `diagonalizable`, this distortion is fully described by anisotropic scaling, which affects the "volume" of the object.

How can one determine the scale factors of the transformation implied by multiplication with a matrix? One must first obtain some notion of the effect of a linear transformation on the volume of an object - this is achieved by the notion of a `determinant`, which can be viewed as a quantification of its "volume". A loose but intuitive definition:

#### Definition 3.2.1 - Determinant: Geometric View

The determinant of a dxd matrix is the (signed) volume of the d-dimensional parallelpiped defined by its row (or column) vectors

This definition is self-consistent because the volume defined by the row vectors and the volume defined by the column vectors of a square matrix can be mathematically shown to be the same

- `determinant` of A = `det(A)`
- `sign` - the sign of the determinant
  - tells us the effect on the `orientation` ob the basis system

Ex. a Householder reflection matrix always has a determinant of -1 becuase it changes the orientation of the vectors it tansforms

Noteworthy: multiplying a nx2 data matrix containing the 2-dimensional scatter plot of a right hand (in its rows) with a 2x2 reflection matrix will change the scatter plot to that of a left hand. The sign of the determinant keeps track of this orentation effect of the linear transformation.

The geometric view of useful becuase it provides us an intuitive idea of what the determinant actually computes in terms of absolute values.

Consider two matrices

100
010
001

and

100
110
001

The determinants of both matrices can be shown to be 1, and both parallelipeds have a base area of 1 and a height of 1

The first is I. I is orthogonal. Orthogonal matrix always forms a unit hypercube, so the absolute value of the determinant is always 1

Ex. a 3x3 matrix of rank 2 - all 3 row vectors must lie on 2d plane. Therefore, no possibility of being 3d, so no volume, so no determinant

The determinant of the dxd matrix A can also be defined in terms of (d-1)x(d-1) submatrices of A

#### Definition 3.2.2 - Determinant: Recursive View

Let A = [a_{ij}] be a dxd matrix and A_{ij} be the (d-1)x(d-1) matrix formed by dropping the ith row and jth column, while maintaining the relative ordering of retained rows and columns. The determinant det(A) is recursively defined as:

1. If A is a 1x1 matrix, its determinant is equal to the single scalar inside of it

2. If A is > 1x1 matrix, its determinant is given by the following for any fixed value of j \in {1..d}:

$\det{(A)} = \sum_{i=1}^{d}(-1)^{(i+j)} a_{ij}\det{(A_{ij})}  $ [Fixed column j]

This fixes a column j, and then expands using all the elements of that column. Any choice of j will yield the same determinant. It is also possible to fix a row i and expand along that row:

$\det{(A)} = \sum_{j=1}^{d}(-1)^{(i+j)}a_{ij}\det({A_{ij}})  $ [Fixed row i]

The recursive definition implies that some matrices have easily computable determinants:

- Diagonal matrix:
  - The determinant of a diagonal matrix is the product of its diagonal entries
- Triangular matrix:
  - The determinant of a triangular matrix is the product of its diagonal entries
  - A matrix containing a row (or columns) of 0s will have a determinant of 0

Consider:

ab
cd

det(A) = ad - bc

```txt
abc   a**   *bc   *bc
def   *ef   d**   *ef
ghi   *hi   *hi   g**
```

= det(A) = a.det[[ef],[hi]] - d.det[[bc],[hi]] + g.det[[bc],[ef]]

= a(ei - hf) - d(bi - hc) + g(bf - ec)

= aei - ahf - dbi + dhc + gbf - gec

its like, for whatever row your operating on, cover values in that row and in the column, and just take the remaining values as matrix

Can observe that the determinant contains 3! = 6 terms, which is the numer of possible ways in which the three elements can be permuted. This perspective provides a permutation-centric definition of the determinant, which is also referred to as the $Leibniz formula$:

#### Definition 3.2.3 - Determinant: Explicit Formula

Consider a dxd matrix A = [a_ij] and let \Sigma be the set of all d! permulations of {1...d}. In other words, for each $\sigma = \sigma_{1}\sigma_{2}...\sigma_{d} \in \Sigma  $, the value $\sigma_{i} $ is a permuted integer from {1...d}. The sign value (denoted by `sgn(\sigma)`) of a permutation $\sigma \in \Sigma  $ is +1, if the permutation can be reached from {1...d} with an even number of element interchanges and it is -1 otherwise. Then, the determinant of A is defined as follows:

$\det{(A)} = \sum_{\sigma \in \Sigma} (sgn(\sigma) \prod_{i=1}^{d}a_{i\sigma_{i}}  )  $

The permutation-centric definition of a determinant is the most direct, although it is difficult to use computationally, and it is not particularly intuitive

#### Problem 3.2.1

Suppose that you have a dxd matrix A, which is not invertible. Provide an informal argument with the geometric view of determinants, as to why addition of i.i.d Gussian noise with variance \lambda to each entry of the matrix A will almost certainly make it invertible

Basically it should shake up the values to the point where they are no longer linearly dependent, and un-collapse the dependent dimensions such that a volume arises

#### Useful Properties of Determinants

The recursive and geometric definitions of the detmerminant imply the following properties:

- 1. Switching two rows (or columns) of a matrix A flips the sign of the determinant
- 2. det(A) = det(A^T)
- 3. A matrix with two identical rows has det(A) = 0
- 4. Multiplying a single row of the matrix A with c to create the new matrix A' results in the multiplication of the determinant of A by c (b/c we are scaling the volume of the matrix parallelepiped by c)
  - det(A') = c.det(A)
  - Muliplying the entire dxd matrix by c scales its determinant by c^d
- 5. det(A) is !=0 only if the matrix is non-singular
  - Geometrically, a parallelepiped of linearly dependent vectors lies in a lower dimensional plane with zero volume

Other important properties

#### Lemma 3.2.1

The determinant of the product of two matrices A and B is the product of their determinants:

det(AB) = det(A).det(B)

Corollary - the det of the inverse of a matrix is the inverse of its determinant:

$\det{(A^{-1})} = \frac{\det{(I)}}{\det{(A)}} = \frac{1}{\det{(A)}}  $

The product-wise property of determinants can be geometrically interpreted in terms of parallelepiped volumes:

- 1. Multiplying matrix A with matrix B (in any order) always scales up the (parallelepiped volume of B) with the volume of A. Therefore, even thought AB != BA (in general), their volumes are always the same
- 2. Multiplying matrix A with a diagonal matrix with values $\lambda_{1}...\lambda_{d}  $ along the diagonal scales up the volume of A with $\lambda_{1}\lambda_{2}...\lambda_{d} $. Not surprising bc we are strethcing the axes with these factors, which explains the nature of the scaling of the volume of the underlying parallelepiped.
- 3. Multiplying A with a rotation matrix simply rotates the parallelepiped, and it does not change the determinant of the matrix
- 4. Reflecting a parallelepiped to its mirror image changes its sign without changing its volum. the sign of the determinant tells us a key fact about the orientation of the data created using multiplicative transformation with A.
  - For ex, consider an nx2 data set D containing the 2-dimensional scatter plot oa right hand in arrows. A negative determinant of a 2x2 matrix A means that multiplicative transformation of the nx2 data set D with A will result in a scatter plot of a right hand in D changing into that of a (possibly stretched and rotated) left hand in DA
- 5. Since all linear transformations are combinations of rotations, reflections, and scaling, one can compute the aboslute effect of a linear transformation on the determinant by focusing on only the scaling portions of the transformation

The product-wise property of determinants is particulary useful for matrices with special structure. For ex, an orthogonal matrix satisfies $A^T A = I$ so therefore we have $\det{(A)}\det{(A^{T})} = det(I) = 1  $. Since the det of A and A^T are equal, it follows that the *square* of the determinant of A is 1

#### Lemma 3.2.2

The determinant of an othogonal matrix is either +1 or -1

One can use this result to simplify the determinant computation of a matrix with various types of decompositions containing orthogonal matrices

#### Problem 3.2.2

Consider a dxd matrix A that is decomposed into the form A = Q\Sigma P^T, where Q and P are dxd orthonormal matrices, and \Sigma is a dxd diagonal matrix containing the nonnegative values \sigma...

- What is the absolute value of the determinant of A?
- Can the sign of the determinant be negative?
- Does the answer change with Q=P?

#### Problem 3.2.3 - Restricted Affine Property of Determinants

ith row becomes the weighted combination of contributions from each A and B.

say like weights of 4, 6 - normalized, these become .4 and .6 of the entire contribution.

- .4 = .4
- .6 = 1 -.4

#### Problem 3.2.4

Work out the determinants of all the elementary row operator matrices from ch1

#### Problem 3.2.6

How can one compute the determinant from the QR decomposition or the LU decomposition of a square matrix

- QR
  - Q is an orthogonal matrix, so det = +-1s
  - R is upper triangular, so det is product of diagonal entries
- LU
  - L is a square lower triangular with ones on the diagonal, so det(L) = 1
  - U is upper triangular, so det is product of diagonal entries

#### Problem 3.2.6

Consider a dxd square matrix A such that A = -A^T. Use properties of determinants to show that if d is odd, then the matrix is singular

#### Problem 3.2.7

Suppose you have a dxd matrix in which the absolute value of every entry is <= 1.

Show that the absolute value of the det <= (d)^{d/2}

Provide an example of a 2x2 matrix in which the determinant is equal to this upper bound (hint: think about the geometric view of determinants)

## 3.3 Diagonalizable Transformations and Eigenvectors

#### Definition 3.3.1 - Eigenvectors and Eigenvalues

A d-dimensional column vector $\bar{x}$ is said to be an `eigenvector` of dxd matrix A, if the following relationship is satisfied for some scalar $\lambda$:

$A\bar{x} = \lambda \bar{x}  $

The scalar $\lambda$ is referred to as its `eigenvalue`

An `eigenvector` can be viewed as "stretching direction" of the matrix, where multiplying the vector with the matrix simply stretches the former. For ex:

- [1, 1]^T and [1, -1]^T are `eigenvectors` of the following matrix
- 3 and -1 are the eigenvalues, respectfully

![alt-text](./3_3_eigenvectors_eigenvalues.PNG)

Each member of the standard basis is an eigenvector of the diagonal matrix, with eigenvalue equal to the ith diagona entry. All vectors are eigenvectors of the identity matrix

The number of eigenvectors of a dxd matrix A may vary, but only diagonalizable matrices represent `anisotropic` scaling in d linearly independent directions; therefore, *we need to be able to find d A linearly independent eigenvectors*.

Let $\bar{v}_{1}...\bar{v}_{d}  $ be d linearly independent eigenvectors and $\lambda_{1}...\lambda_{d}  $ be the corresponding eigenvalues. Therefore, the eigenvector condition holds in each case

$A\bar{v}_{i} = \lambda_{i}\bar{v}_{i}, \forall i \in \{1...d \} $

one can rewrite the condition in matrix form:

$A[\bar{v}_{1}...\bar{v_{d}}] = [\lambda_{1}\bar{v}_{1}...\lambda_{d}\bar{d}] $

By defining V to be a dxd matrix containing v_{1}...v_{d} in its columns, and $\Delta$ to be a diagonal matrix containing $\lambda_{1}...\lambda_{d}  $ along the idaonals, one can rewrite as:

$AV = V\Delta  $

Post-multiplying with V^{-1}, we obtain the diagonalization of the matrix A:

$A = V\Delta V^{-1}  $

- V is an *invertible* dxd matrix containing *linearly independent* eigenvectors
  - aka `basis change matrix`
  - b/c tells us that the linear transformation A is a diagonal matrix $\Delta$ *after changing the basis to the columns of V*
- $\Delta$ is a dxd diagonal matrix, whose diagonal elements contain the `eigenvalues` of A

The `determinant` of a `diagonalizable matrix` is defined by **the product of its eigenvalues**. Since diagonalizable matrices represent linear transforms corresponding to *anisotropic* scaling in arbitrary directions, a diagonalizable transform should scale up the volume of an object by the product of these scaling factors. It is helpful to think of the matrix A in terms of the transform it performs on the unit parallelepiped corresponding to the orthonormal columns of the identity matrix:

$A = AI  $

The transformation scales this unit parallelepiped with scaling factors $\lambda_{1}...\lambda_{d}  $ in d directions. The ith scaling multiplies the volume of the parallelepiped by $\lambda_{i} $. A a result, the final volume of the parallelepiped defined by the I matrix (after all the scalings) is the product of $\lambda_{1}...\lambda_{d} $. This intuition provides:

#### Lemma 3.3.1

The determinant of a diagonalizable matrix is equal to the product of its eigenvalues

The presence of a zero eigenvalue implies that the matrix A is singular because its determinant is zero. One can also infer this fact from the observation that the corresopnding eigenvector v satisfies Av = \bar{0}. In other words, the matrix A is not of full rank because its null space is nonempty. A nonsingular, diagonalizable matrix can be inverted easily according to:

$(V\Delta V^{-1})^{-1} = V \Delta^{-1}V^{-1} $

Note that the $\Delta^{-1}  $ can be obtained by replacing each eigenvalue in the diagonal of \Delta with its reciprocal. Matrices with zero eigenvalues cannot be inverted; the reciprocal of zero is not defined

#### Problem 3.3.1

Let A be a square, diagonalizable matrix. Consider a situation in which we add \alpha to each diagonal entry of A to create A'. Show that A' has the same eigenvectors as A, and its eigenvalues are related to A by a difference of \alpha

It is noteworthy that the ith eigenvector $\bar{v}_{i}$ belongs to the (right) null space of $A - \lambda_{i}I $ because $(A - \lambda_{i}I)\bar{v}_{i} = 0  $. This polynomial expression that yields the eigenvalue roots is referred to as the `charactersitc polynomial of A`

#### Definition 3.3.2 - Characteristic Polynomial

The characteristic polynomial of a dxd matrix A is the degree-d polynomial in \lambda obtained by expanding $\det{(A - \lambda I)}  $

Note that this is a degree-d polynomial, which always has d roots (including repeated or complex roots). The d roots of the characteristic polynomial of *any* dxd matrix are its eigenvalues

#### Observation 3.3.1

The characteristic polynomial f(\lambda) of a dxd matrix A is a polynomial in the following form, where $\lambda_{1}...\lambda_{d}  $ are `eigenvalues` of A:

$\det{(A - \lambda I)} = (\lambda_{1} - \lambda)(\lambda_{2} - \lambda)...(\lambda_{d} - \lambda)  $

Therefore, the eigenvalues and eigenvectors of a matrix A can be computed as follows:

- 1. The eigenvalues of A can be computed by
  - expanding $\det{(A - \lambda I)}  $ as a polynomial expression in \lambda
  - setting it to zero
  - solving for \lambda
- 2. For each root \lambda_{i} of the polynomial, we solve the system of equations (A - \lambda_{i} I)\bar{v} = 0 in order to obtain one or more eigenvectors. The linearly independent eigenvectors with eigenvalue \lambda_{i}, therefore, define a basis of the right null space of (A - \lambda_{i} I)

The charactersitic polynomial of the dxd identity matrix is $(1 - \lambda)^{d}  $. This is consistent with the fact that an identity matrix has d repeated eigenvalues of 1, and every d-dimensional vector is an eigenvector belonging to the null space of A - \lambda I. As another example:

![alt-text](./3_3_eigenvectors_eigenvalues_2.PNG)

- The determinant: $(1 - \lambda)^{2} - 4 = \lambda^{2} - 2\lambda - 3  $
  - equivalent to: $(3 - \lambda)(-1 - \lambda)  $
  - Setting to 0, we get eigenvalues of:
    - 3, -1
- The corresponding eigenvectors are:
  - [1, 1]^T, [1, -1]^T
  - which can be obtained from the null spaces of each $(A - \lambda_{i} I)  $

We need to diagonalize B as $V\Delta V^{-1}  $. The matrix V can be constructed by stacking the eigenvectors in columns

The normalization of columns is not unique, although choosing V to have unit columns (which results in V^{-1} having unit rows) is a common practice. Can construct the diagonalization $B = V \Delta V^{-1}  $ as follows:

![alt-text](./3_3_eigenvectors_eigenvalues_3.PNG)

#### Problem 3.3.2

#### Problem 3.3.3

One can compute a polynomial of a square matrix A in the same way as one computes the polynomial of a scalar - the main differences are that non-zero powers of the scalar are replaced with powers of A and that the scalar term c in the polynomial is replaced by cI. When one computes the characteristic polynomial in thers of its matrix, one always obtains the zero matrix - this is the `Cayley-Hamilton Theorem` and is true for all matrices whether they are diagonalizable or not

#### Lemma 3.3.2 - Cayley-Hamilton Theorem

Let A be any matrix with characteristic polynomial $f(\lambda) = \det{(A - \lambda I)}  $. Then, f(A) evaluates to the zero matrix

The Cayley-Hamilton theorem is true in general for any square matrix A, but it can be proved more easily in some special cases. For ex, when A is diagonalizable, it is easy to show the following for any polynomial fn f():

$f(A) = V f(\Delta)V^{-1}  $

Applying a polynomial fn to a diagonal matrix is equivalent to applying a polynomial fn to each diagonal entry (eigenvalue). Applying the characteristic polynomial to an eigenvalue will yield 0. therefore, f(\Delta) is a zero matrix, which implies that f(A) is a zero matrix. One interesting consequence of the Cayley-Hamilton theorem is that the inverse of a non-zero singular matrix can always be expressed as a polynomial of degree (d-1)!

#### Lemma 3.3.3 - Polynomial Representation of Matrix Inverse

The inverse of an invertible dxd matrix A can be expressed as a polynomial of A of degree at most (d-1)

The constant term in the characteristic polynomial is the product of the eigenvalues, which is non-zero in  the case of nonsingular matrices. Therefore, only in the case of nonsingular matrices, we can write the Cayley-Hamilton matrix polynomial in some form:

$f(A) = A[g(A)] + cI  $ where g(A) is a matrix polynomial of degree (d-1)

Rearranging, we get:

$A[-g(A)/c] = I  $ where $[-g(A)/c] = A^{-1}$

#### Problem 3.3.4

AA = A^2

So A[g(A)] where g(A) is a d-1 polynomial is equivalent

The above lemma explains why the inverse shows many special properties (e.g., commutativity of multiplication with inverse) shown by matrix polynomials. Similarly, both polynomials and inverses of triangular matrices are triangular. Triangular matrices contain eigenvalues on the main diagonal

#### Lemma 3.3.4

Let A be a dxd triangular matrix. Then, the entries $\lambda_{1}...\lambda_{d}  $ on its main diagonal are its eigenvalues.
