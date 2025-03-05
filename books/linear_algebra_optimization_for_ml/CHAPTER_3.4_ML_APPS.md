# 3.4 Machine Learning and Optimization Applications

Creating a little fork for this. Some interesting nuggets and the core chapter notes are too verbose, but this is out of place in a dictionary

## 3.4.1 Fast Matrix Operations in Machine Learning

- normally:
  - computing A^k is expensive
  - sometimes impossible as k goes to \inf limit
- with LA considerations
  - $A^k = V \Delta^{k} V^{-1} $
  - as k -> inf, A^k will either vanish to 0 or explode
    - depends on whether largest eigenvalue is <1 or >1
- where common:
  - working with adjacency matrices of graphs

## 3.4.2 Examples of Diagonalizable Matrices in ML

These are `positive semidefinite` matrices which arise repeatedly in ML

### `Dot Product Similarity Matrix`

- of an nxd data matrix D
- becomes nxn matrix of:
  - pairwise dot products between rows of A
- $S = DD^T $
  - of form of `Gram matrix`
  - must be `positive semidefinite`
- an alternative way of specifying the dataset
  - one can recover D from S
  - one of the most commonon methods is via `eigendecomposition`:
    - $A = Q \Delta Q^T $
    - \Delta contains nonnegative `eigenvalues` (of the `positive semidefinite`) similarity matrix
    - $S = Q\Sigma^2 Q^{T} = (Q\Sigma)(Q\Sigma)^T $
      - where $D' = Q \Sigma $ - a possibly rotated and reflected version of D
- on dimensions:
  - S is nxn matrix
    - data is d dimensional
    - d << n
  - what are the (n-d) dimensions?
    - these correspond to 0s aka `dummy coordinates` - `eigenvalues` of 0
    - b/c rank is at most d

Note: Dot Product is but one similarity function

#### `Gaussian Kernel` - Similarity Matrix

- $Similarity(\bar{x}, \bar{y}) = \exp{\frac{-\|\bar{x} - \bar{y}\|^{2}}{\sigma^2}} $
  - \sigma controls sensitivity of the fn to distances b/w points
- using this, may not be able to recover the original dataset - but will get something similar
  - interestingly, the recovered representations Q\Sigma may be more useful for ML than the original D
  - `nonlinear feature engineering`

### `Covariance Matrix`

computes the (scaled) dot products between *columns* of D AFTER mean-centering the matrix

C = D^T D / N

- `variance`
  - $\sigma^2 = \frac{1}{N}\sum(x - \mu) $
  - where $\mu $ is mean of sample
  - where N is num samples
- `standard deviation`
  - $\sigma = \sqrt{\sigma^2} $
- `covariance`
  - $Cov(X, Y) \frac{1}{N}\sum (x_i - \mu_{X})(y_i - \mu_{Y}) $

`mean-centering` makes \mu_x = \mu_y = 0, so

- $\sigma_{xy} = \frac{\sum_{i=1}^{n} x_{i}y_{i}}{n} $

useful for `principal component analysis`, can be diagonalized as:

C = P \Delta P^T

Relationship:

- `similarity matrix`
  - dot product between rows
  - Gram matrix of the row space of D
  - positive semi-definite
- `covariance matrix`
  - dot product between cols
  - positive semi-definite

#### `Scatter matrix`

- unscaled covariance matrix
- Gram matrix of the column space of D
- positive semi-definite

C = D^T D  (no /N)

#### Decorrelated Data

- $C = P \Delta P^T $
- then, $D' = DP $
- now, $\Delta = P^T C P $
- About \Delta
  - diagonal entries are variances of the individual dimensions of the transformed data
  - represente nonnegative eigenvalues of positive semidefinite matrix C
  - typically, only a few diagonal entries are large
    - others can be dropped from transformed rep
    - >> $P_{k}$

On P_{k}:

- k << d
- transformed data matrix defined as $D_{k}^{'} = D P_{k} $
  - each row is new k-dimensional representation of data
- despite highly reduced dimensionality, still retains most variability
- the discarded (d - k) cols are not very informative

## 3.4.3 Symmetric Matrices in Quadratic Optimization

Many ML problems posed as optimization over squared obj fn, which are quadratic. The process of solving such problem is called `quadratic programming`. This is useful because arbitrary functions can be locally approx as quadratic functions using `Taylor expansion`, which leads to `Newton method`

for $\bar{x}^{T} A \bar{x} $

- A is positive semidefinite
  - `convex function` (bowl with a min and no max)
- A is negative semidefinite
  - `concave function` (upsidedown bowl)
- A is neither positive nor negative semidefinite (aka `indefinite`)
  - have saddle points (no global maxima or minima)
  - saddle points look like both maxima and minima depending on how approaching it

Convex function w/ single global minimum at (0,0)

$f(x_{1}, x_{2}) = x_{1}^{2} + x_{2}^{2} = \begin{bmatrix} x_{1} & x_{2} \end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}x_{1} \\ x_{2} \end{bmatrix} $

Here: $\bar{x}^{T}A\bar{x} = r^{2} $ is the 2x2 identity matrix (which is a trivial form of a positive semidefinite matrix)

Axis-parallel stretching:

If diagonals of A have different values, becomes elliptical (as opposed to perfect circle)

- diagonal entries are **inverse squares** of stretching factors
  - e.g. $\begin{bmatrix} x_{1} & x_{2} \end{bmatrix} \begin{bmatrix}4 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}x_{1} \\ x_{2} \end{bmatrix}$
  - stretches x_{2} instead of x_{1}

Non-axis-parallel stretching:

Consider basis matrix P as a rotation matrix

$\begin{bmatrix}\cos{45} & \sin{45}\\-\sin{45} & \cos{45} \end{bmatrix}  $

Here, use $A = P\Delta P^{T} $ in order to define $\bar{x}^{T}A\bar{x} $

- compute coordinates of $\bar{x} as \bar{y} = P^{T}\bar{x} $
- then compute $f(\bar{x}) = \bar{x}^{T}A\bar{x} = \bar{y}^{T} \Delta \bar{y}  $

Typically, will see form of:

$\begin{bmatrix} x_{1} & x_{2} \end{bmatrix} \begin{bmatrix}4 & -2 \\ -2 & 1\end{bmatrix} \begin{bmatrix}x_{1} \\ x_{2} \end{bmatrix} = 4x_{1}^{2} + x_{2}^{2} - 3x_{1}x_{2} $

- the "$x_{1}x_{2} $" term captures interactions between the attributes x_{1} and x_{2}
  - this is the direct result of a change of basis that is no longer aligned with the axis system

### For Optimal solutions not at (0, 0)

Function of form:

$f(\bar{x}) = (\bar{x} - \bar{b})^{T} A (\bar{x} - \bar{b}) + c $

- where \bar{x} is the location of the optimum (vector)
- c is the optimum value (scalar)
- here, A is equivalent to 1/2 the `Hessian matrix` of the quadratic function
- `Hessian matrix`
  - dxd, H = [h_{ij}]
  - symmetric matrix
  - contains second-order derivatives wrt each pair of variables
  - $h_{ij} = \frac{\partial^{2} f(\bar{x})}{\partial x_{i} \partial x_{j}} $
- $\bar{x}^{T}H\bar{x} $ is the directinoal second derivative of the fn f(\bar{x}) along \bar{x}
  - represents the second derivative of the rate of change of f(\bar{x}) when moving along the direction \bar{x} (acceleration)
  - value is always nonnegative for convex fns irrespective of \bar{x}
  - ensures that the value of f(\bar{x}) is minimum when the first derivative of the rate of change of f(\bar{x}) along each direction \bar{x} is 0
    - aka, Hessian needs to be positive semidefinite
    - which is generalization of $g''(x) \geq 0 $ in 1d convex fns

### General Form of Quadratic Function

$f(\bar{x}) = \bar{x}^{T} A' \bar{x} + \bar{b'}^{T} \bar{x} + c' $

- A' is dxd symmetric matrix
- \bar{b'} is d-dimensional column vector
  - in 1d case, is scalar
- c' is scalar
  - in 1d case, is scalar

For 1d case, we get univariate quadratic function:

$\alpha x^{2} + bx + c $

If \bar{b'} belongs to column space of A', then we can convert the f(\bar{x}) function into `vertex form` - b' being in the column space is **necessary for an optimum to exist**

#### Example

$G(x_{1}, x_{2}) = x_{1}^{2} + x_{2} $

x_{2} is linear so there is not a minimum

`vertex form` considers only strictly quadratic fns in which all cross-sections of the functions are quadratic. Only strictly quadratic fns are interesting for optimization b/c linear functions usually do not have a maximum or minimum

### indefinite functions

When $\bar{x}^{T} A \bar{x} $ is indefinite, and has both positive and negative eigenvalues:

$g(x_{1}, x_{2}) = \begin{bmatrix} x_{1} & x_{2} \end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix} \begin{bmatrix}x_{1} \\ x_{2} \end{bmatrix} = x_{1}^{2} - x_{2}^{2} $

The gradient at (0, 0) is 0, which seems to be an optimum point, however it behaves like both a max and min when looking at second derivatives (saddle point I think)

## 3.4.4 Diagonalization Application: Variable Separation for Optimization

Consider quadratic function $f(\bar{x}) = \bar{x}^{T} A \bar{x} + \bar{b}^{T}\bar{x} + c $

Unless the symmetric matrix A is diagonal, the resulting function contains terms of the form: $x_{i}x_{j} $, these are referred to as `interacting terms`

Most real world quadratic functions contain such terms. However, any multivariate quadratic function can be transformed to an additively separable function (without interacting terms) by basis transformation of the input variables of the function

This change of basis brings us back to linear algebra tricks - additively separable functions are much easier to optimize. This is because one can decompose the optimization problem into smaller optimization problems on individual variables

For example, a multivariate quadratic function would appear as a simple sum of univariate quadratic functions (each of which is extremely simple to optimize)

### Definition 3.4.3 - Additively Separable Functions

A function $F(x_{1}, x_{2}, ..., x_{d}) $ in d varaibles is said to be additively separable, if it can be expressed in the following form for appropriately chosen univariate functions $f_{1}(.), f_{2}(.), ..., f_{d}(.) $:

$F(x_{1}, x_{2}, ..., x_{d}) = \sum_{i=1}^{d} f_{i}(x_{i}) $

Consider following quadratic fn defined on a d-dimensional vector $\bar{x} = [x_{1}, ..., x_{d}]^{T}  $

$f(\bar{x}) = \bar{x}^{T}A\bar{x} + \bar{b}^{T}\bar{x} + c $

Since A is a dxd symmetric matrix, one can diagonalize it as $A = V\Delta V^{T} $, and ise the variable transform $\bar{x} = V\bar{x'} $ (which is the same as $\bar{x'} = V^{T}\bar{x} $)

On performin gthis transformation one obtains the new function $g(\bar{x'}) = f(V\bar{x'}) $, which is identical to the original function in a different basis. It is easy to show that the quadratic function may be expressed as follows:

$f(V\bar{x'}) = \bar{x'}^{T}\Delta\bar{x'} + \bar{b}^{T}V\bar{x'} + c $

After this variable transformation, one obtains an additively separable function, because the matrix \Delta is diagonal. One can solve for \bar{x'} using d univariate optimizations, and then transform back \bar{x'} to \bar{x} using $\bar{x} = V\bar{x'} $

This approach simplifies optimization, but the problem is that eigenvector computation of A can be expensive. However, one can generalize this idea and try to find *any* matrix V (with possibly non-orthogonal columns), which satisfies $A = V\Delta V^{T} $ for some diagonal matrix \Delta. (Note, that A = V \Delta V^{T} would not be a true diagonalization of A if the columns of V are not orthonormal. However, it is good enough to create a separable transformation for optimization, which is what we really care about)

The columns of such non-orthogonal matrices are computatinoally much easier to evaluate than true eigenvectors, and the transformerd variables are referred to as `conjugate directions`. The columns of V are referred to as `A-orthogonal directions`, because for any pair of (distinct) columns $\bar{v}_{i} and \bar{v}_{j} $ we have $\bar{v}_{i}^{T}A\bar{v}_{j} = \Delta_{ij} = 0 $. There are an infinite number of possible ways of creating conjugate directions, and the eigenvectors represent a special case.

In fact, a generalization of the Gram-Schmidt method can be used to find such directions. This basic idea forms the principle of the `conjugate gradient descent` method (seen in Ch5 in future), which can be used for non-quadratic functions. Here, we provide a conceptual overview of the iterative conjugate gradient methods for arbitrary (possibly non-quadratic function) h(\barPx) from the current point $\bar{x} = \bar{x}_{t} $

- 1. Create a quadratic approximation f(\bar{x}) of non-quadratic function h(\bar{x}) using the second-order Taylor expansion of h(\bar{x}) at $\bar{x} = \bar{x}_{t} $
- 2. Compute the optimal solution x* of the quadratic function f(\bar{x}) using the separable variable optimization approach discussed above as a set of d univariate optimization problems
- 3. Set $\bar{x}_{t+1} = \bar{x}* $ and $t \Leftarrow t + t$. Go back to step 1

The approach is iterated to convergence. This provides the conceptual basis for the conjugate gradient method (detailed method is shown in Ch5 5.7.1)

## 3.4.5. Eigenvectors in Norm-Constrained Quadratic Programming

A problem that arises frequently in different types of ML settings is one in which we wish to optimize $\bar{x}^{T} A \bar{x} $ where \bar{x} is constrained to unit norm. Here, A is a dxd `symmetric data matrix`.

This type of problem arises in many feature engineering and dimensionality reduction applications like:

- principal component analysis
- singular value decomposition
- spectral clustering

Such an optimization problem is posed as follows;

- Optimize $\bar{x}^{T} A \bar{x} $
- subject to: $\|\bar{x}\|^{2} = 1 $

The optimization problem can be in either minimization of maximization form. Constraining the vector \bar{x} to be the unit vector fundamentally changes the nature of the optimization problem.

Unlike the previous section, it is no longer important whether the matrix A is positive semidefinite or note. One would have a well defined optimal solution, even if the matrix A is indefinite. Constrainin gthe norm of the vector helps in avoiding vectors with unbounded magnitudes or trivial solutions (like the zero vector), wven when the matrix A is indefinite

Let $\bar{v}_{1}...\bar{v}_{d} $ be the d orthonormal eigenvectors of the symmetric matrix A. Note that the set of eigenvectors creates a basis for R^d, and therefore any d-dimensional vector \bar{x} can be expressed as a linear combination of $\bar{v}_{1}...\bar{v}_{d} $ as:

$\bar{x} = \sum_{i=1}^{d} \alpha_{i}\bar{v}_{i} $

We will re-parameterize this optimization problem in terms of the parameters $\alpha_{1}...\alpha_{d} $ by substituting for \bar{x} in the optimization problem. By making this substitution, and setting each $A\bar{v}_{i} = \lambda_{i}\bar{v}_{i} $, we obtain the following re-parameterized optimization problem:

- Optimize $\sum_{i=1}^{d} \lambda_{i}\alpha_{i}^{2} $
- subject to: $\sum_{i=1}^{d} \alpha_{i}^{2} = 1 $

The expression $\|\bar{x}\|^{2} $ in the constraint is simplified to $(\sum_{i=1}^{d} \alpha_{i}\bar{v}_{i}) \cdot (\sum_{i=1}^{d} \alpha_{i}\bar{v}_{i}) $; we can expand it using the distributive property, and then we use the orthogonality of the eigenvectors to set $\bar{v}_{i}\bar{v}_{j} = 0 $

The objective function value is $\sum_{i} \lambda_{i}\alpha_{i}^{2} $, where the different $\alpha_{i}^{2} $ sum to 1. Clearly the maximum possible values of this objective function are achieved by setting the weight $\alpha_{i}^{2} $ of a single value of \lambda_{i} to 1, which corresponds to the minimum or maximum possible eigenvalue (depending on whether the optimization problem is posed in min or max form)

- The maximum value of the norm-constrained quadratic optimization problem is obtained by setting \bar{x} to the largest eigenvector of A
- The minimum value is obtained by setting \bar{x} to the smallest eigenvector of A

This problem can be generalized to finding a k-dimensional subspace. In other words, we wna tot find orthonormal vectors $\bar{x}_{1}...\bar{x}_{k} $, so that $\sum_{i} \bar{x}_{i}A\bar{x_{i}} $ is optimized (should the first \bar{x}_{i} be ^T ?)

- Optimize $\sum_{i=1}^{k} \bar{x}_{i}^{T} A \bar{x}_{i} $
- subject to:
  - $\|\bar{x}_{i}\|^{2} = 1 \forall i \in \{1...k \} $
  - $\bar{x}_{1}...\bar{x}_{k} $ are mutually orthogonal

The optimal solution to this problem can be derived using a similar procedure. An alternate solution with the use of Lagrangian relaxation in Ch 6.6 is shown. Here, the optimal solution is stated:

- The maximum value of the norm-constrained quadratic optimization problem is obtained by using the largest k eigenvectors of A
- The minimum value is obtained by using the smallest k eigenvectors of A

Intuitively, these results make geometric sense from the perspective of anisotropic scaling cause by symmetric matrices like A. The matrix A distorts the space with scale factors corresponding to the eigenvalues along orthonormal directions corresponding to the eigenvectors. The objective function tries to either max or min the aggregate projections of the distorted vectors $A\bar{x}_{i}$ on the original vectors $\bar{x}_{i}$, which is th esum of the dot products between $\bar{x}_{i} and A\bar{x}_{i} $. By picking the *largest* k eigenvectors (scaling directions), this sum is maximized. On the other hand, by picking the *smallest* k directions, the sum is minimized.
