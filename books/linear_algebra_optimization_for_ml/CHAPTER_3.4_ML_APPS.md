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

C = P

Relationship:

- `similarity matrix`
  - dot product between rows
- `covariance matrix`
  - dot product between cols

#### `Scatter matrix`

- unscaled covariance matrix

C = D^T D  (no /N)
