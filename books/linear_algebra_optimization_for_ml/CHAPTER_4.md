# Chapter 4. Optimization Basics: A Machine Learning View

"If you optimize everything, you will always be unhappy." - Donald Knuth

## 4.1 Introduction

Many ML models are cast as continuous optimization problems in multiple variables

Least-squares regression is foundational to ML:

- in least squares, find the best fit solution to a system of equations
  - system does not need to be consistent
  - if system is consistent, yields a loss of zero
  - loss is aggregate squared error of the best fit
  - target variable is numeric
- foundational to:
  - linear algebra
  - optimization
  - ML
- historically preceded classification problem frame
  - classification models were modifications of least squares regression
  - target variable is discrete (typically binary)
  - optimization model for linear regression needs to be "repaired" to work with discrete target variables

Most continuous optimization methods use differential calculus in one form or another

- main idea (of differential calculus):
  - provide a quantification of the *instantaneous* rate of change of an objective function wrt each of the variables in its argument
  - optimization methods based on this use the fact that the rate of change of an obj fn at a particular set of values of the optmization variables provides hints on how to iteratively change the optimization variable(s) and bring them closer to an optimum solution
  - such iterative algorithms are old, but pretty easy to do once computers were invented

## 4.2 The Basics of Optimization

Optimization problems:

- have an objective function
  - defined in terms of a set of variables = `optimization variables`
- goal is to compute the values of the variables at which the obj fn is either maximized or minimized
  - minimization is common, and referred to as `loss function`
    - `loss function` = obj fun with certain types of properties quantifying a nonnegative "cost" associated with a particular configuration of variables

### 4.2.1 Univariate Optimization

Consider: $f(x) = x^2 - 2x + 3 $

- becomes: $f(x) = (x - 1)^2  + 2$
  - clearly, when (x - 1) = 0, we are at a min
  - so, min value is at x = 1
- or, compute the first derivative wrt x, and set to zero
  - remember, at min value, rate of change wrt x is zero
    - so, the tangent to the plot at that point is horizontal
  - $f'(x) = \frac{d(f(x))}{dx} = 2x - 2 = 0$
  - but this only works for parabolas (x^2)
    - x^3 when f'(x) = 0 is just a saddle b/c s shaped
  - `critical points` = when f'(x) = 0
    - may be max, min, or saddle

Determining if `critical point` is max, min, saddle:

- looking at f''(x):
  - if f''(x) > 0 (is +), then a minima
  - if f''(x) < 0 (is -), then a maxima
  - if f''(x) = 0, then is a `degenerate` critical point
- so:
  - if f'(x) = 0 and f''(x) > 0:
    - is a `local minimum`

`Taylor expansion` can provide some insight:

$f(x_{0} + \Delta) \approx f(x_{0}) + \Delta f'(x_{0}) + \frac{\Delta^{2}}{2}f''(x_{0})  $

- we seek to compare $f(x_{0} + \Delta)$ to $f(x_{0}) $
- where $\Delta $ is:
  - $|\Delta| $ is extremely small value
  - positive or negative
  - $\Delta^2 > 0$ (because square)
- $\Delta f'(x_{0}) = 0 $
- because both $\Delta^{2} $ and $f''(x_{0}) $ are > (is +)
  - then, $f(x_{0} + \Delta) = f(x_{0}) + \epsilon $
    - where $\epsilon $ > 0 (is +)
  - this means that $f(x_{0} < f(x_{0} + \Delta))  $ for any small value of $\Delta $, whether its positive or negative
    - aka $x_{0} $ is a minimum wrt its *immediate locality*

Taylor expansion also provides insights as to why `degenerate` case is problemmatic ($f'(x_{0}) = f''(x_0) = 0 $)

- when $f''(x_0) = 0$:
  - one would need to keep expanding taylor series until reaching first non-zero term
  - if first non-zero term is positive (+), then:
    - $f(x_{0} + \Delta) < f(x_{0}) $
    - ex: $f(x) = x^{4} $ at $x_{0} = 0 $
      - here, x_0 is min wrt immediate locality
      - but first non-zero term is negative, or it depends on the sign of $\Delta$

### Problem 4.2.1

Quadratic functions are easy, with a single min or max depending on the sign of the quadratic term.

Other fns may have multiple turning points. ex: sin(x) is periodic, and has an infinite number of minima/maxima over (-inf, +inf)

Ex:

- $F(x) = (x^4 /4) - (x^3 / 3) - x^2 + 2 $
- $F'(x) = x^3 - x^2 - 2x = x(x + 1)(x -2 ) = 0 $
- $F''(x) = 3x^2 - 2x - 2 $
  - -> local min at x = -1,
    - F''(-1) > 0 (3 + 2 - 2)
    - F(-1) = 1/4 + 1/3 - 1 + 2 = 19/12
  - -> local min at x = 0, F'' < 0 (-2)
  - -> global min at x = 2,
    - F''(2) > 0 (12 - 4 - 2)
    - F(2) = 4 - 8/3 - 4 + 1 = -2/3 **global minimum**

### Problem 4.2.2

### Problem 4.2.3

```python
# Coefficients of F'(x) = 4x^3 -24x^2 +42x -22
coefficients = [4, -24, 42, -22]

# Find the roots
roots = np.roots(coefficients)
```

#### 4.2.1.1 Why We Need Gradient Descent

Solving f'(x) = 0 for x provides an `analytical` solution for a `critical point`, but it is not always possible to solve analytically

- `analytical solution`
  - any solution derived from algebraic or calculus-based manipulations
    - relying on identities, symbolic simplifications, derivatives, integrals, etc
    - exact expressions, not approximations
    - "solving by hand with math rules"
    - uses infinite, limits, series, or special functions
    - `closed-form solution`
      - strict subset of analytical solutions
      - x = -b +- sqrt(b^2 - 4ac) / 2a
- `numerical solutions`
  - approximate the answer using a computational method

Gradient descent - populat approach for optimizing objective functions (irrispective of functional form)

- start with an initial point $x = x_0 $
- successively update x using the steepest descent direction:
  - $x = \Longleftarrow x - \alpha f'(x) $
  - where a > 0 regulates step size, aka `learning rate`
- in univariate problems, x only has two directions of movement:
  - increase x or decrease x
  - one direction causes ascent, the other causes descent
- in multivariate problems, there is an infinite number of directions
  - the generalization of the notion of univariate detivative leads to steepest descent direction
- value of x changes in each iteration by $\delta x = -\alpha f'(x) $
  - when learning rate is "infentesimally" small, the above update will always reduce f(x):
    - $f(x + \delta x) \approx f(x) + \delta x f'(x) = f(x) - \alpha[f'(x)]^2 < f(x) $
  - small values of learning rate (\alpha > 0) is not advisable b/c convergence will take a while
  - too large \alpha makes effect of update unpredictable (b/x computed gradient is no longer good approximation)
    - remember: gradient is only instantaneous rate of change
  - extremely large values, can cause solution to *diverge*, exploding to a terminating numerical overflow

#### 4.2.1.2 Convergence of Gradient Descent

Execution of gradient-descent updates will generally result in a sequence of values: x_0, x_1, ..., x_t of the optimization variable, which has become successively closer to an optimum solution

- as x_t nears optimum, f'(x_t) tends to be closer and closer to zero
  - aka: the absolute step size will tend to reduce over execution of algorithm
- `monotonically`
  - consistent increases, or decreases
- two cases:
  - convergence (good)
  - divergence (bad)

#### 4.2.1.3 The Divergence Problem

- basically, things see-saw from one side of a quadratic function to another, because the large overshoot flips the sign of the gradient
- and, unless learning rate is dramatically reduced, the exploding gradient continues to increase in magnitude while alternating sign

Telltale sign of divergence:

- size of parameter vector seems to increase rapidly
- optimization objective worsens (at the same time)

First adjustment is to lower initial learning rate

Other literature discusses step size selection in depth

### 4.2.2 Bivariate Optimization

Helpful to bridge gap in complexity from single variable optimization to multivariate optimization

- $F(x, y) = f(x) + f(y) = x^2 + y^2 - 2x - 2y + 6 $
- $G(x, y) = g(x) + g(y) = ([x^4 + y^4]/4) - ([x^3 + y^3]/3) - x^2 - y^2 + 4 $

Note, both above are `additively separable`:

- when have multivariate fn x^2 + y^2
- but there are no xy terms - no `interacting terms`
- all quadratic functions can be represented in additively separable form

F(x) is easy because only one local (and thus global) minimum

G(x) is hard bc multiple minima, only one of which is global. To find this, need to compute the `partial derivative` of the obj functions F(x, y) and G(x, y) in order to perform gradient descent.

- `partial derivative`
  - comptues derivative wrt a particular variable
  - while treating other variables as contants
- `gradient`
  - a vector of `partial derivatives`

Gradient of $F(x, y) = x^2 + y^2 - 2x - 2y + 6 $ as:

$\nabla F(x,y) = \begin{bmatrix} \frac{\partial F(x,y)}{\partial x} & \frac{\partial F(x,y)}{\partial y} \end{bmatrix}^{T} = \begin{bmatrix}2x - 2 \\ 2y - 2 \end{bmatrix} $

so like if were taking the partial of x and hold all others (y) constant, we just ignore y terms which do not include x

Also note that $\nabla_{x, y} g(x,y) $ denotes choice of variables wrt which the gradient is computed

Once finding the gradient, just set to zero like f'(x) and solve for [x, y] = [1, 1] (for the above equation)

However, this wil **NOT** always lead to a system of equations with a closed form solution. The common solution is ot use gradient-descent updates wrt the optimization variablex [x, y] as follows:

$\begin{bmatrix}x_{t+1}\\ y_{t+1}\end{bmatrix} \Longleftarrow \begin{bmatrix}x_{t}\\ y_{t}\end{bmatrix} - \alpha \nabla g(x_{t}, y_{t}) = \begin{bmatrix}x_{t}\\ y_{t}\end{bmatrix} - \alpha \begin{bmatrix}2x_{t} - 2\\ 2y_{t} - 2\end{bmatrix}  $

Now, consider more complecated fn (that is not additively separable):

$H(x, y) = x^2 - \sin(xy) + y^2 - 2x $ here, sin(xy) ensures not additively separable

$\nabla H(x, y) = \begin{bmatrix}\frac{\partial H(x, y)}{\partial x} & \frac{\partial H(x, y)}{\partial y} \end{bmatrix}^{T} = \begin{bmatrix}2x - y\cos(xy) - 2 \end{bmatrix} \\ 2y - x\cos(xy) $

Despite partial derivative components not being expressed in terms of individual variables, gradient descent updates can be performed in similar manner

Looking at:

$\nabla G(x, y) = \begin{bmatrix}x^3 - x^2 - 2x \\ y^3 - y^2 - 2y \end{bmatrix} = \bar{0} $

We get 9 pairs of $(x, y) \in \{-1, 0, 2 \} \cross \{-1, 0, 2 \} $ which satisfy first order optimality conditions, and therefore all 9 are critical points. Among these there are:

- single global minimum
- three local minima
- single local maximum at (0, 0)
- the other four are saddle points

The classification of points as minima, maxima, or saddle points can only be accomplished with the use of a multivariate second-order conditions.

Pay attention to the rapid proliferation of the number of possible critical points satisfying the optimality conditions (when optimization problem contains two variables instead of one). In general, when a multivariate problem is posed as a sum of univariate functions, the **number of local optima can proliferate exponentially fast** with the number of optimization variables

#### Problem 4.2.4

### 4.2.3 Multivariate Optimization

Most ML problems are defined on large parameter space containing multiple optimization variables. Variables of optimization problem are parameters used to create a `prediction function` of either observed or hidden attributes of the ML problem

In linear regression, optimization variables w_1, w_2, ... w_d are used to predict the dependent variable y from the independent variables x_1, ..., x_d:

$y = \sum_{i=1}^{d} w_{i} x_{i} $

Going forward, assume the notation "w_1, ..., w_d" represent `optimization variables` (weights) as opposed to other "variables":

- x_i
- y

Both of which are observed values from the dataset at hand (which are constants from the optimization perspective)

The objective fn often penalize differences in observed an predicted values of specific attributes, such as y

Ex. if ew have many observed tupes of the form: [x_1, x_2, ..., x_d, **y**], one can sum up the values of $(y - \sum_{i=1}^{d} w_{i} x_{i})^{2} $ over all observed tuples. Such objective fns are referred to as `loss functions`

- `loss function` = `objective function`
  - `loss function` = $J(\bar{w})$
    - function of a vector of multiple optimization variables
    - $\bar{w} = [w_{1} ... w_{d}]^T $
      - (assume is a column vector unless explicitly specified)
    - here, use "w_1...w_d" for optimization variables
      - because $\bar{X}, x_{i}, \bar{y}, and y_{i} $ are reserved for attributes of data (data whose values are observed)
      - *d* corresponds to the number of optimization variables
    - use "J" because im guessing this is a `Jacobian` under the hood
      - ith component of d-dimensional gradient (vector) is the partial derivative of J wrt the ith parameter w_i

Simplest approach to solve the optimization problem directly (without gradient descent) is to set gradient (vector) to zero, leading to:

$\frac{\partial J(\bar{w})}{\partial w_{i}} = 0, \forall i \in \{1...d \} $

These conditions lead to a system of d equations, which can be solved to determine parameters w_1...w_d. As in univariate, would like to have a way to characterize whetehr a critical point (i.e., zero-gradient point) is a max, min, or infelction >> leads to second order condition

Second order condition: for single-variable, for f(w) to be min, f''(w) > 0. For multivariate, we generalize to the `Hessian` matrix.

`Hessian` - instead of a *scalar* second derivative, we have dxd matrix of second-derivatives, including the **pairwise** derivatives of J wrt different pairs of variables. The `Hessian` of the loss function $J(\bar{w}) $ wrt w_1...w_d is given by dxd *symmetric* matrix `H`, in which the (i, j)th entry H_{ij}:

$H_{ij} = \frac{\partial^{2}J(\bar{w})}{\partial w_{i} \partial w_{j}} $

Note, the (i, j)th entry of the Hessian is equal to the (j, i)th (yes backward from first instance) entry because partial derivatives are commutative according to `Schwarz's theorem`. Being symmetric is useful fact for many computational algorithms that require eigendecomposition of the matrix

The `Hessian` is a direct generalization of the univariate second derivative f''(w). For a univariate fn, the Hessian is a 1x1 matrix containing f''(w) as its only entry. Strictly speaking, the Hessian is a *function* of \bar{w}, and it should be denoted by $H(\bar{w}) $, though we denote it by H for brevity.

In the event that the fn J(w) is quadratic, the entries in the Hessian do not depend on the parameter vector $\bar{w} = [w_{1}...w_{d}]^T $. This is similar to the univariate case, where the second derivative f''(w) is a constant when the fn f(w) is quadratic (like f(x) = x^2 as the highest power). In general, the Hessian depends on the value of the parameter vector \bar{w} at which it is computed

!!! important !!!

For a parameter vector \bar{w} at which the gradient is zero (critical point), one needs to test the Hessian matrix H the same way we test f''(w) in univariate fns. Needs to be **positive-definite** for a point to be guaranteed to be a minimum (like f''(w) needs to be positive > 0 (+))

To illustrate, consider second-order, multivariate Taylor expansion of J(\bar{w}) in the immediate locality of \bar{w}_{0} along the direction \bar{v} and small radius \epsilon > 0:

$J(\bar{w_{0}}, + \epsilon \bar{v}) \approx J(\bar{w_{0}}) + \epsilon \bar{v}^{T} [\nabla J(\bar{w}_{0})] + \frac{\epsilon^{2}}{2}[\bar{v}^{T} H \bar{v}] $ where $[\nabla J(\bar{w}_{0})] = 0$ because f'(x) for critical point criterion

The Hessian depends on parameter vector (remember $H(\bar{w})$ ), is computed at $\bar{w} = \bar{w}_{0} $. It is evident that `objective fn` $J(\bar{w}_{0}) < J(\bar{w}_{0} + \epsilon \bar{v}) $ when we have $\bar{v}^{T} H \bar{v} > 0 $. If we can find even a single direction \bar{v} where we have $\bar{v}^{T} H \bar{v} < 0 $, then \bar{w} is clearly not a minimum wrt its immediate locality.

!!! important !!!

A matrix H that satisfies $\bar{v} H \bar{v} > 0 $ is positive definite

The notion of positive definiteness of the Hessian is the direct generalization of the second-derivative condition f''(w) > 0 for univariate fns

Assuming that the gradient is zero at critical point \bar{w}, we can summarize the following second-order optimality conditions:

- 1. If Hessian is positive definite at $\bar{w} = [w_{1}...w_{d}]^{T} $, then $\bar{w} $ is a `local minimum`
- 2. If the Hessian is negative, then \bar{w} is a local maximum
- 3. If the Hessian is indefinite (x^T H(w) x = 0), then a saddle point
- 4. If Hessian is positive- or negative **semi**-definite, then the test is inconclusive

These conditions represent direct generalizations of univariate optimality conditions. Helpful to examine what a saddle point for an indefinite Hessian looks like. Consider:

$g(w_{1}, w_{2}) = w_{1}^{2} - w_{2}^{2} $

The Hessian of this quadratic function is independent of the parameter vector [w_1, w_2]^T, and is defined as follows:

$H = \begin{bmatrix}2 & 0 \\ 0 & -2 \end{bmatrix} $

This Hessian turns out to be a diagonal matrix, which is clearly indefinite b/c one of the two diagonal entries is negative. [0, 0] is a critical point bC the gradient is zero at that point. However, is a saddle point because of the indefinite nature of the Hessian

#### Problem 4.2.5

det < 0 = indefinite = saddle point

- det(H) > 0 and tr(A) > 0 (or f_xx > 0) >> positive definite (minima)
- det(H) > 0 and tr(A) < 0 (or f_xx < 0) >> negative definite (maxima)
- det(H) < 0 >> indefinite (one positive, one negative eigenvalue) (saddle point)
- det(H) = 0 >> inconclusive

Setting the gradient of the obj fn to 0 and then solving the resulting system of equations is usually computationally difficult. Therefore, gradient-descent is used - here with learning rate \alpha:

$[w_{1}...w_{d}]^{T} \Longleftarrow [w_{1}...w_{d}]^{T} - \alpha \begin{bmatrix} \frac{\partial J(\bar{w})}{\partial w_{1}} ... \frac{\partial J(\bar{w})}{\partial w_{d}} \end{bmatrix}^{T} $

Can rewrite the above in terms of the gradient of the obj fn wrt \bar{w}:

$\bar{w} \Longleftarrow \bar{w} - \alpha \nabla J(\bar{w}) $

Here, $\nabla J(\bar{w}) $ is a column vector containing the $\partial $ partial derivatives of $J(\bar{w}) $ wrt different parameters in the column vector \bar{w}. \alpha usually varies over the course of the algorithm
