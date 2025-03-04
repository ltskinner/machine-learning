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
