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
  - useful for finding closest approximation of n dimensional \bar{b} on a plane defined by < n vectors *when the point does not lie on the plane*
    - (isnt in the column space (space spanned by the columns))
  - $P_{cols} = A(A^T A)^{-1} A^T$
    - note this is the `left-inverse` and projects columns
    - also note that we use the real values of A, after finding which cols are the basis cols
    - do NOT use the RREF format column literal values
    - See [GRAM_PROJECTION_MATRIX](./GRAM_PROJECTION_MATRIX.md)
  - $P_{rows} = B^T(BB^T)^{-1} B $
    - this is the `right-inverse` and projects rows
  - $\bar{b}' = A\bar{x} = P\bar{b}  $
  - also, $P = QQ^T$ in context of QR decomp, where A = QR
  - QR is also nice b/c no need to compute inverse
  - the `projection matrix` P *only* depends on the vector space spanned by the columns of A
  - $P = AA^{+} $ (from Moore-Penrose) is also a form of projection matrix
    - here, $AA^{+} $ is both `symmetric` and `idempotent`
    - see [similarties to left `Gram matrix`](./GRAM_PROJECTION_MATRIX.md)
- `Q` - `orthogonal matrix`
  - in many cases is a basis
  - QR decomposition
  - A matrix where its transpose is its inverse
  - $A^{T} = A^{-1} $
  - $AA^T = AA^{-1} = I $
  - the product of any number of orthogonal matrices is always an orthogonal matrix
- `R`
  - upper triangular
  - rotation matrix
  - $\begin{bmatrix}1 & 0 & 0 \\0 & -1 & 0 \\0 & 0 & -1\end{bmatrix} $
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
  - only values on the diagonal
  - $RAC = \Delta$
    - R = nxn row operations (to row echelon form)
    - C = dxd col operations (to diagonal form)
    - $\Delta$ is nxd rectangular diagonal matrix
  - multiplication with $\Delta$ scales the:
    - rows of A, if $\Delta A = \lambda_{i}a_{ij} $
    - cols of A, if $A\Delta = a_{ij} \lambda_{j} $
- $\Lambda$ - `eigenvalues matrix`
- `J` (optimization)
- `S` - `similarity matrix`
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
  - can also be expressed using Gram matrix $S = A^T A$
    - $<\bar{x}, \bar{y}> = \bar{x}^T A \bar{y} $
- `norm` or `euclidean norm`
  - $\|x\|^{2} = x \cdot x = \sum_{i}^{d}x_{i}^{2} = x^T x $
  - taking sqrt of this is `euclidean distance/length from origin`
- `cross products`
  - give you a new `vector`
  - i  j  k
  - a1 a2 a3
  - b1 b2 b2
  - - i - j + k
- `orthogonal`
  - if `dot product` is 0 aka the angle bw is 90deg
- `orthonormal`
  - each pair of vectors is mutually orthogonal
  - norm of each vector is 1 (unit vectors)
- `projection`
  - a new set of coordinates wrt a new changed set of directions
  - "the 1-dim projection operation of a vector x on a unit vector is the dot product"
  - x is modified to point *in the direction of the unit vector*
- `standard basis representation` or `atd basis representation`
  - requires
    - a `coordinate`
    - a `basis`
  - `coordinate` indecies should align with each vector in basis
  - simple Qv = b
  - to *find* a `coordinate` *from* a `std basis representation`
  - Qv = b
    - where Q is the basis
    - b is the `std basis representation`
    - solve for v (RREF to x1 + x2 = 3, x2 = 2)

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
- col vector: Ux
- row vector: yU^{t} (for same matrix U)
- cols of data matrix D: UD or V_{col}D^{T}
- rows of data matrix D: DU^{T} or DV_{row}

### `linear transformation`

Core linear transform:

- $f(c\bar{x}) = c \cdot f(\bar{x})  $
- $f(\bar{x} + \bar{y}) = f(\bar{x}) + f(\bar{y})  $

Aka:

- scalar multiplication of a vector
- addition of two vectors
- rotations, reflections, scalings of x

In context of Matrix multiplication:

- $Ax$ acts like $cx$, making it a linear transform

### `translation` (as a special case of `affine transforms`)

**not** a linear transform

- $f(\bar{x}) = \bar{x} + \bar{y}  $
- THIS IS DIFFERENT THAN:
  - $f(\bar{x} + \bar{y}) = f(\bar{x}) + f(\bar{y})  $

#### `Ax` `linear transform` specifically

- the Ax operation
- transforms from a d-dimensional space to an n-dimensional space
- transforms to a n-dimensional space from a d-dimensional space
  -  A   x
  - nxd dx1 --> nx1
  - each row contains values that span the columns
- Ax is effectively a **weighted sum** of the columns of A
  - so x is the weights, corresponding to each column
  - for each row, the weight is applied to the value for each column
- $A\bar{x} = \sum_{i=1}^{d} x_{i}\bar{a}_{i} = \bar{b} $
  - each x_{i} corresponds to the weight of the ith direction a_{i}
    - which is the ith coordinate of b
  - scaling factor of directions. Each index of b corresponds to a direction
- `solution set`
  - three cases (3 cases) arise finding the solution to b
  - also see notes on [Normal Equation Ax = b for ML](../../MACHINE_LEARNING.md)
  - 1. b does **NOT** exist in column space of A
    - so no solution exists
    - however, *best fits* are possible
    - common for `over-determined systems of linear equations`
      - "number of rows is much greater than number of cols" - "tall"
    - typical for inconsistent systems of equations (trailing rows of 0s)
  - 2. b does exist in column space of A
    - A must have linearly independent columns
    - the solution is unique
      - each column has a leading value of 1 for the corresponding row
    - when A is square, solution is: $\bar{x} = A^{-1}\bar{b}  $
  - 3. b occurs in the column space of A
    - columns of A are linearly dependent
    - there is an infinite number of solutions
      - there are "free columns" where there is no leading value of 1 for the corresponding row
    - common where number of cols is greater than number of rows - "wide"
    - see [2.8 An Optimization-Centric View of Linear System](./CHAPTER_2.md)
      - for tall matrices w/ linearly independent cols:
        - most concise solution:
        - $x = (A^T A)^{-1} A^T b  $ (uses left inverse)
      - for wide matrices w/ linearly independent rows:
        - most concise solution
        - $x = A^{T}(AA^{T})^{-1}b $ (uses right inverse)
  - *all zero (rows) in A' need to be matched with zero entries in b' for the system to have a solution*
  - more advanced:
    - consider: $BA\bar{x} = B\bar{b} $
    - A introduces a set of intermediate constraints...
    - Has potential to "trim" by:
      - discarding unused inputs (to A)
      - preventing reaching unreachable outputs (from A)

### `affine transforms`

the combination of a `linear transformation` with a `translation`:

$f(\bar{x}) + A\bar{x} + \bar{c} $

the translation (b in AX + b) translates the origin from V to a new point in W

formally (READ THIS IN BROWSER):

$f(\lambda\bar{x}) + [1 - \lambda]\bar{y} = \lambda f(\bar{x}) + [1 - \lambda]f(\bar{y})  $

linear transforms are a subset of affine transforms

### Terms in this book

- `affine transform`
  - see above
  - is a generalization of linear transforms
  - allow translations (so b != 0)
- `linear transform` = `linear operator`
  - special case of affine transform where b = 0
  - T: V -> W
  - maps between two vector spaces V -> W
    - preserves additivity
    - preserves scalar multiplication
    - maps the origin of V to the origin of W (does not allow translations)
  - T(x) = Ax (where no constant vector is added)
  - `linear operator`
    - special case where V = W
    - the transformation maps a vector space to itself
    - T: V -> V
- `linear function`
  - when translation is allowed
  - f(x) = Ax + b
    - where b != 0 is a translation vector
    - so the origins to not map to each other

### Vector and Matrix Multiplication

A multiplication of a `vector` and a `matrix` is some combination of:

- rotation - rigid
- scaling - NOT rigid
- reflection - rigid

Applied to **the vector**

(the vector is "the thing", and the matrix contains the information about what to do with the thing)

### Vector Spaces

- `linear independence`
  - for a set of non-zero (non-trivial) scalar coeffs [x_1, ..., x_{d}]
  - $\sum_{i=1}^{d} x_{i}\bar{a}_{i} != \bar{0}$
  - basically, theres no coeffient that can be applied to any combination of vectors that would result in the sum of those vectors being 0
    - if we can show the sum of coefficients *does* = 0, then we prove it is *not* linearly independent by contradiction
  - means $[\bar{a}_{i}...\bar_{a}_{d}]$ are `mutually orthogonal`
  - linearly independent square matrices are invertible
  - [http://thejuniverse.org/PUBLIC/LinearAlgebra/LOLA/indep/examples.html](http://thejuniverse.org/PUBLIC/LinearAlgebra/LOLA/indep/examples.html)
    - Two vectors from the origin are NOT parallel (overlapping)
    - If are parallel, each is just a scaling of the other
    - In 3d space, if three vectors lie in a plane, they are NOT independent
      - the 3rd vector can be computed as a linear combination of other 2
    - 2d:
      - v = 2u (scaling)
    - 3d:
      - v = 2u + 4w (scaling, but each)
      - v = [u,w][2, 4]^T  
- `basis`
  - a minimal set of vectors so that all vectors can be expressed as linear combinations
    - so $\bar{v} = \sum_{i=1}^{d} x_{i}\bar{a}_{i}  $
  - *The vectors in a basis must be linearly independent*
  - is a **maximal** set of `linearly independent` vectors in it
  - non-orthogonal basis systems are a pain to deal with
  - orthogonal basis are pog
- `normal equation` - see [Original Notes](./NORMAL_EQUATION.md)
  - in Ax = b
    - x represents the `scaling factors` for each independent direction contributed by the columns of A
    - x only has meaning in the context of A
    - A contains structural and geometric significance
    - A contains a basis
      - if linearly independent, A is the basis
      - if not
        - RREF to identify `pivot columns`
        - use corresponding original cols from A as the basis
  - the vector $\bar{b} - A\bar{x} (= \bar{0})  $
    - joins $\bar{b}$ to its closest approximation:
    - $\bar{b}' = A\bar{x}'  $ on the hyperplane defined by the columns space of A
    - (which is orthogonal to the hyperplane and therefore every col of A)
  - bringing us to the `normal equation`
    - $A^T (\bar{b} - A\bar{x}) = \bar{0}  $, which yields
    - $\bar{x} = (A^{T}A)^{-1} A^{T}\bar{b} $
    - (assuming A^T A is invertible - easy to do when A is tall)
    - see `Projection Matrix` above for more detail
- `projection formula - 1d`
  - simplification of the normal equation for when the projection "matrix" is a 1d basis (vector)
  - from form b = Vc
  - $c = \frac{v^T b}{v^T v}$
  - lineage:
    - $A^{T}(A\bar{x} - \bar{v}) = \bar{0}  $
    - becomes
    - $\bar{x} = (A^{T}A)^{-1}$A^{T}\bar{v}  $
    - for 1d matrices (vectors), (A^{T}A)^{-1} becomes 1/(A^{T}A)^{-1}
  - we say:
    - $c = Proj_{v}(b) = \frac{v^T b}{v^T v} $
    - read as the direction of v that b covers/aligns with
      - the "shadow" of b onto direction v
    - the fractional part is the *magnitude* of b in the direction of v
    - multiplying this magnitude * b gives you the actual projection vector
- `dimensionality` (of a vector space)
  - The number of members in every possible basis set of a vector space V is always the same
- `transform` form:
  - $\bar{x}_{b} = P_{a -> b} \bar{x}_{a}$
  - transform to b from a using P
    - P performs the transform from a to b
    - P projects
      - from the d-dimensional subspace (column space)
      - into the n-dimensional subspace (row space)
      - this is why the column space is a subspace of R^n
    - rank
      - the dimensionality of the largest subspace reachable starting from either the row or the column space
  - lineage:
    - $A\bar{x}_{a} = B\bar{x}_{b} = \bar{x}  $
    - reorganizing:
    - $\bar{x}_{b} = [B^{-1}A]\bar{x}_{a}  $ where $P_{a -> b} = [B^{-1}A] $
  - for non-square non-invertible:
    - $\bar{x}_{b} = (B^{T}B)^{-1} B^{T}A \bar{x}_{a}  $
- `span`
  - the vector space defined by all possible linear combinations of the vectors in a set
  - (like the C-Space I feel like)
  - the dimension of the span is the number of linearly independent vectors
- `spans`
  - means a set of vectors spans a space
    - so every vector in the space can be written as a linear combination of vectors
- `disjoint vector space`
  - if two spaces do not contain any vector in common (other than the zero vector)
  - disjoint pairs of vector spaces do NOT need to be `orthogonal`
- `orthogonal vector spaces`
  - if for any pair of vectors from each space, their dot product is zero
    - e.g. $\bar{x} \cdot \bar{w} = 0  $
  - orthogonal vector spaces are always `disjoint`

### Spaces

Very high level description:

- for matrix A we have a set of columns
- if columns are linearly independent
  - each column defines a unique/independent direction/`dimension` in the space
- `column space` is the space `spanned` by the combination of these independent columns
- these are all equal:
  - = the number of `linearly independent` columns
  - = number of independent directions `spanned`
  - = `dimension` of the `column space`
  - = the `rank` of the `columns space`
- `dimensionality` of the `column space` = number of `rows`
  - defines the `ambient space` where the columns live
- the `dimensionality` of each column:
  - contains n entries corresponding to the number of rows
  - this makes the `column space` a `subspace` of R^n

Other definitions:

- `columns space` aka `column rank`
  - vector space spanning columns of nxd matrix A
  - subspace of R^n (the row dimensionality)
- `row space` aka `row rank`
  - vector space spanning rows of nxd matrix A
    - or, space spanning cols of dxn matrix A^T
  - subspace of R^d (the col dimensionality)
- `matrix rank` = rank of `col space` = rank of `row space`
  - min rank(n, d)
  - see [Chapter 2 search 'min' or 'Corollary 2.6.1'](./CHAPTER_2.md)
- [see Four Fundamental Subspaces](./FOUR_FUNDAMENTAL_SUBSPACES.md)
- `(right) null space` aka `row null space`
  - for Ax = 0
  - the space unreachable by the rows of A
  - any vector in Null(A) is orthogonal to every row of A
- `left null space` aka `col null space`
  - for A^T x = 0
  - the space unreachable by the cols of A (or the rows of A^T)
  - any vector in Null(A^T) is orthogonal to every row of A^T
- for square and non-singular (invertible) matrices, the null space only contains the zero vector
- `full rank`
  - full rank = linearly independent
  - `positive-definite` full rank square matrices have an empty null space
    - full rank matrices must be `invertible`
  - where the rows of a *square matrix* must be linearly independent when the columns are linearly independent, these matrices are of `full rank`
  - for *rectangular matrices* to be full rank, *either* the rows or columns are linearly independent
    - `full row rank`
      - when rows are linearly independent
    - `full column rank`
      - when columns are linearly independent
- `row equivalence` and `column equivalence`
  - two matrices are row/col equivalent if one matrix is obtained from the other by a sequence of elementary row/column operations (interchange, addition, multiplication)
  - applying row/column operations does not change the vector space spanned by the rows/columns of the matrix
    - these operations do not fundamentally change the (normalized) row/col set of the matrix
- `row echelon form`
  - upper-triangular matrix
  - convenient row-equivalent conversion of matrix A
  - useful for solving linear systems of the type $A\bar{x} = \bar{b} $
  - by applying the same row operations to matrix A and vector $\bar{b}$ in the system of equations, A can be simplified to a form that is easily solvable
  - equivalent to `Gaussian elimination` method for solving systems of equations
- `Gaussian elimination`
  - Official order:
    - row addition operations
      - perform on lower of the two rows
    - row interchange operations
    - row scaling operations
  - useful for finding a basis set of a bunch of (possibly linearly dependent) vectors
- `back substitution`
  - when we have form Ax = b, and A and b are known
  - 1 0 0 | 4
  - 0 1 0 | 6
  - 0 0 1 | 5
  - for back substitution, sometimes we need to scale items on left by items on right to get to 1
  - then, the numbers on right become solution
`parametric form`

$x = \begin{bmatrix}x_1 \\ x_2 \\ x_{3}\end{bmatrix} = a \begin{bmatrix}1 \\ 1\\ 0\end{bmatrix} + b \begin{bmatrix}-1 \\ 0\\ 1\end{bmatrix} $

based on:

- $x_{1} - x_{2} + x_{3} $
- $x_{2} = a$ only parameterize free variables
- $x_{3} = b$ only parameterize free variables

based on:

$\begin{bmatrix}1 & -1 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0\end{bmatrix}$

### Matrix Properties

- `inconsistent`
  - refers to a system of equations Ax = b, not JUST A
  - only **AFTER converting to RREF** format can we inspect for rows of 0s
    - properties of A alone that hint at an inconsistent system:
      - A has fewer linearly independent columns than rows
        - rank(A) < n, making the column space smaller
      - A has fewer columns than rows
      - A already posessing rows of 0s
  - have no solution bc a zero value on the left is equated with a non-zero value on the right
    - like zeros in RREF A but non zero in same row of b
    - *all zeros in A' need to be matched with zero entries in b' for the system to have a solution*
    - simple: b does not lie in the span of columns of A
  - inconsistent systems where neither the rows nor cols are linearly independent mean that neither A^T A or AA^T is invertible
    - which, sees use in L (left-inverse) and R (right-inverse) definitions
- `strictly positive` matrix
  - only positive and non-zero values
- `positive semidefinite`
  - iff all its eigenvalues are non-negative
  - **not** guaranteed to be invertible
  - Any matrix of the form BB^T or B^T B (i.e. Gram matrix form)
- `positive definite` matrix
  - only positive and zero values
  - matrix A cannot be singular
  - iff all its eigenvalues are non-negative
  - guaranteed to be invertible
- `negative semidefinite matrices`
  - every eigenvalue is non-positive
  - can be converted into a positive semidefinite matric by reversing the sign
- `negative definite`
  - which every eigenvalue is strictly negative
- `indefinite`
  - symmetric matrices with both *positive and negative* eigenvalues
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
  - invertibility implies linearly independent cols (and/or linearly independent rows)
- `left inverse`
  - $L = (A^T A)^{-1}A^T  $
    - because $LA = (A^T A)^{-1} A^T A = I_{d}  $
  - `regularized left inverse`
    - $\bar{x} = (A^T A + \lambda I_{d})^{-1}A^T \bar{b}  $
  - **same** as `right inverse` if square matrix lol
- `right inverse`
  - $R = A^T(AA^T)^{-1}  $
    - because $AR = AA^T(AA^T)^{-1} = I_{n}  $
  - `regularized right inverse`
    - $\bar{x} = A^T(AA^T + \lambda I_{n})^{-1} \bar{b}  $
    - this comes from opimization focused approach to addressing inconsistent linear systems, where the obj fn is:
      - $J = \|A\bar{x} - \bar{b} \|^2 + \lambda \sum_{i=1}^{d}x_{i}^{2}  $
      -        Best Fit term              Conciseness term
      - lambda is the regularization term, which favors small absolute components of the vector \bar{x}
      - note that (AA^T + \lambda I_{n}) is always invertible
        - it is the \lambda I_{n} term that guarantees invertiblity when AA^T or A^T A is inconsisent and lacking linearly independent rows AND columns
    - if the primary goal is best fit, let \lambda be very small
- `Moore-Penrose pseudoinverse`
  - use for rectangular matrices - denoted as A^{+} (instead of A^{-1})
  - see left/right inverse above
  - $\lim_{\lambda \rightarrow 0^{+}} (A^T A + \lambda_{d})^{-1} A^T = \lim_{\lambda \rightarrow 0^{+}} A^T(AA^T + \lambda I_{n})^{-1} $
  - When A is invertible, all inverses are the same
    - conventional inverse
    - left inverse
    - right inverse
    - Moore-Penrose pseudoinverse
  - When only columns of A are linearly independent
    - MP is left inverse
  - When only rows of A are linearly independent
    - MP is right inverse
  - When neither rows nor columns of A are linearly independent
    - MP provides a generalized inverse that none other can provide
  - to compute:
    - have A = QR from QR decomposition
    - MP = $A^{+} = \lim_{\lambda \rightarrow 0^{+}} (R^T R + \lambda I_{d})^{-1} R^T Q^T = \lim_{\lambda \rightarrow 0^{+}} R^T(RR^T + \lambda I_{n})^{-1} Q^T = R^T(RR^T)^{-1} Q^T  $
- `inverting singular matrices` - `matrix inversion lemma`
  - Neumann Series:
    - $(I + A)^{-1} = I - A + A^2 - A^3 + A^4 + ... + $ Infinite Terms
    - $(I - A)^{-1} = I + A + A^2 + A^3 + A^4 + ... + $ Infinite Terms
  - Sherman-Morrison-Woodburry Identity:
    - $(A + UV^{T})^{-1} = A^{-1} - A^{-1}U(I + V^{T}A^{-1}U)^{-1}V^{T}A^{-1}  $
    - Full:
      - $(A + UCV^{T})^{-1} = A^{-1} - A^{-1}U(C^{-1} + V^{T}A^{-1}U)^{-1}V^{T}A^{-1} $
      - same except for C, which is the I
- `diagonal matrix`
  - only values on the diagonal
  - multiplication with $\Delta$ scales the:
    - rows of A, if $\Delta A = \lambda_{i}a_{ij} $
    - cols of A, if $A\Delta = a_{ij} \lambda_{j} $
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
- `skew symmetric matrix`
  - square matrix
  - A^T = -A
  - A = -A^T
  - these are like 90 degree rotation matrices, but work better in higher dimensions
  - the diagonal entries of A are always 0
  - det(A) = det(-A^T) = (-1)^{d}det(A)
    - when d is odd, (-1)^{d} = -1
    - so det(A) = -det(A) ==> det(A)+det(A) = 0
    - or 2det(A) = 0 = det(A)
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
  - P^{k} = P
- `idempotent property` of `projection matrices`:
  - $P^2 = P = (QQ^{T})(QQ^{T}) = Q(Q^{T}Q)Q^{T} = QQ^{T} $ - note (Q^{T}Q) is an identity
- `indefinite`
  - symmetric matrices with both positive and negative eigenvalues
- `energy`
  - another name for the squared Frobenius norm
  - The energy of a rectangular matrix A is equal to the trace of either AA^{\top} or A^{\top}A
    - $\|A\|_{F}^{2} = Energy(A) = tr(AA^{\top}) = tr(A^{\top}A)  $
- `trace`
  - tr(A) of a square matrix is defined by the sum of its diagonal entries
  - tr(A) = \sum_{i=1}^{n} a_{ii}^{2}
  - tr(A) is equal to the sum of the eigenvalues, whether it is diagonalizable or not
- `Gram matrix` - [see Gram Matrix](./GRAM_PROJECTION_MATRIX.md)
  - `(right) Gram matrix`
    - $A^T A$ of the column space of A
    - the cols of A are linearly independent iff A^T A is invertible
  - `left Gram matrix`
    - $AA^T$ of the row space of A
- [`Gram-Schmidt / QR Decomposition`](./GRAM_SCHMIDT_QR_LU_DECOMP.md)
- [`LU Decomposition`](./GRAM_SCHMIDT_QR_LU_DECOMP.md)
- `Singular Value Decomposition` aka `SVD`
  - square matrix only
  - $A = U \Delta V^{\top}  $
    - $U $ defines basis for `column space` - has orthonormal cols
      - col space of UDV is subspace of col space of U
    - $V $ defines basis for `row space` - has orthonormal cols
      - row space of UDV is subspace of row space of V
    - $\Delta$ is a *nonnegative* scaling matrix
  - all linear transformations defined by matrix multiplication:
    - can be expressed as a sequence of rotations/reflections
    - with a single ansiotripic scaling factor

## Push Through Identity

### For Inverses

Core:

$A(BA)^{-1} = B^{-1}(A^-1) $ b/c $(AB)^{-1} = B^{-1}A^{-1}  $

Alternative:

- $A^{-1}BC^{-1} = (AC)^{-1}(ABC)  $
- $(I + AB)^{-1}A = A(I + BA)^{-1}  $
- $(I + AB)^{-1} = I - AB(I + AB)^{-1} = I - A(I + BA)^{-1}B  $

Inverse of a Sum

$(A + B)^{-1} = A^{-1} - A^{-1}(AB^{-1} + I)^{-1}  $

### For Transposes

$A^{T}(AA^{T})^{m} = (A^{T}A)^{m}A^{T}  $ where m is any non-negative integer

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
  - G_{col}(2, 4, \alpha)\bar{x} = \bar{x}^{T}G_{row}(2, 4, \alpha)
    - pre multiply vs post multiply
- `Householder reflection matrix`
  - `orthogonal matrix` that reflects a vector x into any "mirror" hyperplane of arbitrary orientation
  - v **must** be normalized
  - $H = (I - 2\bar{v}\bar{v}^{\top})$ is an `elementary reflection matrix`
    - $H = (I - 2\bar{v}\bar{v}^{\top}/\|v\|^{2})$ (explicit normalization)
  - $H = Q_{1}Q_{1}^T - Q_{2}Q_{2}^{T}  $
    - theres some weird shit where:
    - $(I - 2\bar{v}\bar{v}^{\top}) = Q_{1}Q_{1}^T - Q_{2}Q_{2}^{T} $
    - but lets see if this rears its head again...
      - in this context, $Q_{2} = \|v\|$
      - and $H = (I - 2Q_{2}Q_{2}^{\top})  $
  - $\bar{x}' = H\bar{x}  $

## [EigenX](./EIGENX.md)

- `determinant`
  - the "volume" `scaling factor` of a matrix
  - product of `eigenvalues` $\det{(A)} \prod_{i} \lambda_{i}  $
  - of `diagonal matrix`
    - det = product of diagonal entries
  - of `triangular matrix`
    - det = product of diagonal entries
  - of matrix with row (or col) of 0s
    - det = 0
  - of `orthogonal matrix`
    - det = 1 or -1
- `eigenvectors` (right by default)
  - d different directions (linearly independent)
  - right $A\bar{x} = \lambda \bar{x} $
  - left $\bar{y}A = \lambda \bar{y} $
  - eigenvectors point in the directions that remain unchanged under transformation
  - they have a property where when A is multiplied against them, they result in the original vector just being scaled by some scalar value - the `eigenvalue` $\lambda$
  - solving for eigenvectors from eigevnalues:
    - solve (Q - \lambda_{1}I)v = 0
    - (do for each eigenvalue you have)
- `eigenvalues`
  - d different scale factors
  - the value an eigenvector is scaled by when A is multiplied against it
  - --> sum is Trace
  - --> product is Determinant
- `diagonalizable matrix`
  - special type of `linear operator`
  - simultaneously scales along d different directions (`eigenvectors`) with d different scale factors (`eigenvalues`)
  - $A = V\Delta V^{-1}  $
    - $V $ contains d `eigenvectors`
    - $\Delta $ contains d `eigenvalues`
  - Matrices containing repeated eigenvalues and missing eigenvectors of the repeated eigenvalues are *not diagonalizable*
  - I think all symmetric, orthogonal matrices are diagonal matrices
    - symmetric = A = A^T
    - orthogonal = A^T = A^-1
    - diagonal = all zeros except the diagonal
    - so, A = A^-1
- `defective matrix`
  - a matrix where a `diagonalization` *does not exist*
  - missing eigendirections (eigenvectors) that contribute distinctly (not linearly independent)
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

## Polynomial Function

$f(A) = \sum_{i=0}^{k} c_{i}A^{i}$

$f(A) = c_{0}I + c_{1}A + c_{2}A^{2} + ... + c_{k}A^{k}$

### Cayley-Hamilton

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

DV_{r} will rotate each **row** of D

$ V_{row} =
\begin{bmatrix}
\cos{\theta} & \sin{\theta} \\
-\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

V_{c}D^{T} will rotate each **column** of D

$ V_{col} =
\begin{bmatrix}
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta}
\end{bmatrix}
$

### 3x3 Rotation Matrices

$\begin{bmatrix}1 & 0 & 0 \\ 0 & \cos(\theta) & -\sin(\theta) \\ 0 & \sin(\theta) & \cos(\theta)\end{bmatrix} $

Here, the eigenvalues of all 3x3 rotation matrices are:

$[1, e^{i\theta}, e^{-i\theta} ]  $

### Reflection - rigid

reflect across X-axis

2d:

$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$

3d:

$
\begin{bmatrix}
1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & -1
\end{bmatrix}
$

remember - the axis being rotated about is left unchanged, but the other axis need to change

### Scale - NOT rigid

- same between pre- and post-
- aka:
  - `dilation`
  - `contraction`
- `anisotropic` - when scaling factors across different dimensions are different

scale x and y by factors of c_{1} and c_{2}

$
\begin{bmatrix}
c_{1} & 0 \\
0 & c_{2}
\end{bmatrix}
$

### Interchange - rigid

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

### Scaling - not RIGID

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
