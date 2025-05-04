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

Following univariate pattern, above >>

$J(\bar{w_{0}}, + \epsilon \bar{v}) \approx J(\bar{w_{0}}) + \frac{\epsilon^{2}}{2}[\bar{v}^{T} H \bar{v}] > J(\bar{w_{0}})$ when $\bar{v}^T H\bar{v} > 0$ (positive definite)

and

$J(\bar{w_{0}}, + \epsilon \bar{v}) \approx J(\bar{w_{0}}) + (-)\frac{\epsilon^{2}}{2}[\bar{v}^{T} H \bar{v}] < J(\bar{w_{0}})$ when $\bar{v}^T H\bar{v} < 0$ (negative definite)

(note - free handed the above two equations, not from book)

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

## 4.3 Convex Objective Functions

The presence of local minima creates uncertainty about the effectiveness of gradient-descent algorithms. Ideally, one would like to have an objective fn w/o local minima. Objective functions with this property is in the class of `convex functions`. `Convex functions` are defined only with domains that that are convex, using convex sets

### Definition 4.3.1 - Convex Set

A set S is convex, if for every pair of points $\bar{w}_{1}, \bar{w}_{2} \in S $ the point $\lambda \bar{w}_{1} + [1 - \lambda]\bar{w}_{2} $ must also be in S for all $\lambda \in (0, 1) $

Aka, it is impossible to find a pair of points in the set, such that any of the points on the straight line joining them do not lie in the set.

A `closed convex set` is one in which the boundary points (i.e. limit points) of the set are included in the set [-2, +2]

An `open convex set` is one in which all points within the boundary are included, but not the bounary itself (-2, +2)

![alt text](./4_2_convex_sets.PNG)

A `convex function` $F(\bar{w}) $ is defined as: a function with a convex domain that satisfies the following conditions for any $\lambda \in (0,1) $:

$F(\lambda \bar{w}_{1} + (1 - \lambda) \bar{w}_{2}) \leq \lambda F(\bar{w}_{1}) + (1 - \lambda)F(\bar{w}_{2}) $

- convex if a >= 0
- strictly convex if a > 0
- concave if a <= 0
- strictly concave if a < 0

Also note that the convex sets you can visualize with circles. However, convex functions, in the 2d case, are parabolic U in an x y chart. so as long as *a* is non negative, will be convex. This allows for non-smootheness (aka "kinks") in the line However, this only works for univariate. for multivariate we have to look at Hessian and all that and require positive definiteness to assess convexity.

Also, this inequality is something we can do with univariate to build intuition, but we could also go directly to the F''(w) > 0 condition. this scales into multivariate with the v^T H v > 0 condition, also guaranteeing convexity. Note that the above F <= F inequality is just grounded in the literal definition of convexity

One can generalize the convexity condition to k points, discussed in problem 4.3.1

#### Problem 4.3.1

Im guessing that the function domain eventually has something to do with the span of the domain, and there are lacking directions pointed at making parts of the domain unreachable and thus non-convex

Attaching to the domain of the Function, like in abstract space we *can* draw a line between two points, which "cuts the corner", but the function cannot "cut the corner" because it doesnt point in the directions of the corner, so it is forced to take a longer route (cant shortcut), which is how we get the inequality and a function of each of the individual parts is > the function of the \lambda w1 + (1 - \lambda) w2 vector

The simplest example of a convex objective function is the class of quadratic functinos in which the leading (quadratic) term has a **non-negative coefficient**:

$f(w) = aw^2 + bw + c$

*a* needs to be nonnegative for the fn to be considered quadratic. All linear functions are always convex, because the convexity property holds with equality

#### Lemma 4.3.1

A linear function of the vector $\bar{w}$ is always convex

Convex functions have a number of useful properties that are leveraged in practical applications

#### Lemma 4.3.2

Convex functions obey the following properties

- 1. The sum of convex functions is always convex
- 2. The maximum of convex functions is convex
- 3. The square of a nonnegative convex function is convex
- 4. If F(.) is a convex function with a single argument and G(\bar{w}) is a linear function with a scalar output, then F(G(\bar{w})) is convex
- 5. If F(.) is a convex non-increasing function and G(\bar{w}) is a concave function with a scalar output, then F(G(\bar{w})) is convex
  - this means f'(x) is <= 0
  - like a -log(x) slope that at limit becomes a flat line
    - note it must be negative log (-log) for 1st and 2nd derivative properties
  - which bruh: this means we can take a concave function and "broadcast" it into a concave domain with log(f(x)) and optimize in that domain
    - **this is literally Cross-Entropy** as a convex objective
- 6. If F(.) is a convex non-decreasing function and G(\bar{w}) is a convex function with a scalar output, then F(G(\bar{w})) is convex
  - means that f'(x) >= 0
  - common in:
    - regularization terms
    - loss fns
    - confidence penalties
  - ex. x^2 (squaring, as we showed in 3.)

#### Problem 4.3.2

Prove all the results of Lemma 4.3.2 using the definition of convexity

The fact that every local minimum is a global minimum can be characterized geometrically (aka `first-derivative condition` aka `first derivative condition`): The entire convex function will always lie above a tangent to a convex function

An important consequence of convexity is that one is often guaranteed to reach a global optimum if successful convergence occurs durring the gradient-descent procedure

![alt text](./4_7_convex_function_above_any_tangent.PNG)

This tangent condition can be written algebraically using the gradient of the convex function at a given point w_0. This provides an alternate definition of convexity:

#### Lemma 4.3.3 - First-Derivative Characterization of Convexity

A differentiable function $F(\bar{w}) $ is a convex function if and only if the following is true for any pair $\bar{w}_{0} $ and $\bar{w} $:

$F(\bar{w}) \geq F(\bar{w}_{0}) + [\nabla F(\bar{w}_{0})] \cdot (\bar{w} - \bar{w}_{0}) $

Note that if the gradient of F(w) = 0 at $\bar{w} = \bar{w}_{0} $, it would imply that $F(\bar{w}) \geq F(\bar{w}_{0}) $ for any $\bar{w} $ aka: $\bar{w}_{0} is a global minimum $

therefore, any critical point that satisfies the first-derivative condition is a global minimum. Disadvantage with this approach is it applies to only differentiable functions like:

- abs()
- ReLU
- Hinge Loss

Characterizations:

- graph has sharp corner/kink
- functions involve max, min, abs
- piecewise function (esp. linear functions)
- left and right derivative dont match

#### Lemma 4.3.4 - Second-Derivative Characterization of Convexity

The twice differentiable function $F(\bar{w})$ is convex, if and only if it has a positive semidefinite Hessian at every value of the parameter $\bar{w} $ in the domain of F(.)

This method has a disadvantage of requiring F(w) to be twice differentiable. Therefore, the following convexity definitions are equivalent for twice-differentiable functions defined over R^d:

- 1. Direct:
  - The convexity condition:
  - $F(\lambda \bar{w}_{1} + [1 - \lambda]\bar{w}_{2}) \leq \lambda F(\bar{w}_{1}) + (1 - \lambda)F(\bar{w}_{2}) $
  - is satisfied for all $\bar{w}_{1}, \bar{w}_{2} $ and $\lambda \in (0, 1) $
- 2. First-derivative:
  - The first-derivative condition
  - $F(\bar{w}) > F(\bar{w}_{0}) + [\nabla F(\bar{w}_{0})] \cdot (\bar{w} - \bar{w}_{0} ) $ is satisfied for all $\bar{w} $ and $\bar{w}_{0} $
- 3. Second-derivative:
  - The Hessian of $F(\bar{w})$ is positive semi-definite for all $\bar{w} $
  - aka the second derivative > 0 (no need for the v^T H v stuff here)
    - literally just analytically derive F''

One can use any of the conditions as the definition of convexity, and then derive the other two as lemmas. However, the direct definition is more general (bc no dependency on differentiability). It is often the case that a particular definition is easier to use than another

!!! important !!!

Many ML objective fns are of the form: $F(G(\bar{w})) $ where $G(\bar{w}) $ is the linear function $\bar{w} \cdot \bar{X}^{T} $ for a row vector containing a d-dimensional data point $\bar{X} $, and F(.) is a univariate function.

In such a case, one only needs to prove that the univariate function F(.) is convex, based on the the final portion of Lemma 4.3.2. It is particularly easy to use the second-order condition $F''(.) \geq 0 $ for univariate functions

Next problems showcase convexity of the **logarithmic logistic loss function**. This function is useful for showing the convexity of logistic regression

#### Problem 4.3.4

literally, find second derivative and assess if its >= 0 or not to determine convexity

#### Problem 4.3.5

`strict convexity` is a slightly stronger condition, were we just replace the >= signs with >

Example: a bowl with a flat bottom is convex, but not strictly convex. **A strictly convex function will have a unique global minimum**

#### Lemma 4.3.5

A strictly convex function can at most contain one critical point. If such a point exists, it will be the global minimum of the structly convex function

The above property is easy to show by using either the direct or first-order definition of strict convexity.

!!! important !!!

One often constructs objective functions in ML by adding convex and strictly convex functions. In such cases, the sum of these functions is strictly convex

#### lemma 4.3.6

The sum of a convex function and a strictly convex function is strictly convex

Many objective fns in ML are convex, and they can often be made strictly convex by adding a **strictly convex regularizer**

A special case of convex functions is that of quadratic convex functions, which can be directly expressed in terms of the positive semidefinite Hessian. Although the Hessian of a function depends on the value of the parameter vector at a specific point, it is a constant matrix in the case of quadratic functions - example:

$f(\bar{w}) = \frac{1}{2}[\bar{w} - \bar{b}]^{T}H[\bar{w} - \bar{b}] + c $

Here, $\bar{b} $ is a d-dimensional column vector and *c* is scalar

A convex objective function is an ideal setting for a gradient-descent algorithm; the approach will never get stuck in a local minimum. Although the objective functions in comples ML models (like NNs) are not convex, they are often close to convex. As a result, gradient-descent methods work quite well in spite of the presence of local optima

For any convex function $F(\bar{w}) $, the region of space bounded by $F(\bar{w}) \leq b $ for any constant *b* can be shown to be a convex set. This type of constraint is encountered often in optimization problems. Such problems are easier to solve because of the convexity of the space in which one wants to search for the parameter vector.

Read: I think b is some arbitrary constraint we put in. is it learned or like a 0-1 regularized term?

## 4.4 The Minutiae of Gradient Descent

Such as selecting learning rate \alpha can have huge consequence in both good and bad way

### 4.4.1 Checking Gradient Correctness with Finite Differences

Many ML models use complex obj fns over millions of parameters. Gradients are computed analytically and hand-coded into the alg, or computed using automatic differentiation methods in applications like NNs. In all cases, analytical or coding errors remain a real possibility, which may not be obvious during execution. Knowing the reason for poor performance of an anlgorithm is a critical step in deciding whether to simply debug the alg or to make fundamental design changes

Consider situation where we compute gradient of obj fn: $J(\bar{w}) = J(w_{1}...w_{d}) $. In the finite-difference method, we sample a few of the optimization parameters from w_1...w_d and check their partial derivatives using `finite-difference approximation`.

- `finite-difference approximation`
  - perturb an optimization parameter w_i by a small ammount \Delta
  - and approximate the partial derivative wrt w_i using the difference between the perturbed value of the obj fn and the original value

$\frac{\partial J(\bar{w})}{\partial w_{i}} = \frac{J(w_{1}..., w_{i} + \Delta, ..., w_{d}) - J(w_{1}..., w_{i}, ..., w_{d})}{\Delta} $

- This way of estimating gradient referred to as finite-difference approximation
- one will not obtain an exact value of the partial derivative
  - however, in cases where gradients are computed incorrectly, the value of a `finite-difference approx` is often **so wildly different** from the analytical valuu that the error becomes self evident
- Typcially, it suffices to check the pds of small subset of the params to detect systematic problem in gradient computation

### 4.4.2 Learning Rate Decay and Bold Driver

A constant learning rate often poses a dilemma to the analyst.

- a lower learning rate used early on will cause the alg to take too long to reach anywhere close to an optimal solution
- a large learning rate will sllow the algorithm to come reasonably close to a good solution at first, but the alg will then oscillate around the point for a very long time

Allowing the learning rate to decay over time can naturally achieve the desired learning rate adjustment to avoid these challanges

$\alpha_{t} $ is the decaying learning rate, and subscripted with *t* for the time stamp

$\bar{w} \Longleftarrow \bar{w} - \alpha_{t} \nabla J $

Time *t* is typicall ymeasured in terms of the number of cycles over all training points. The most common decay functions are:

- Exponential decay:
  - $\alpha_{t} = \alpha_{0} \exp(-k \cdot t) $
- Inverse decay:
  - $\alpha = \frac{\alpha_{0}}{1 + k\cdot t} $
- Step decay
  - Rate reduced by a particular factor every few steps of gradient descent

Parameter k controls the rate of decay

Another popular approach for adjusting the lr is the `bold-driver algorithm`. Here, the LR changes depending on whether the obj fn is improving or worsening

- lr increased by a factor of ~5% in each iteration as long as steps improve
- as soon as worsen in a step, the step is *undone* and an attempt is made again with lr reduced by factor of ~50% (yes 50)
- continue until convergence
- note:
  - does not work in some noisy settings of gradient descent, where obj fn is approximated by using samples of the data.
  - examples of such a noisy setting:
    - `stochastic gradient descent`
  - here, test obj fn and adjust lr after m steps, rather than a single step
  - change in obj fn can be measured more robustly across multiple steps
  - and all m steps must be undone when the obj fn worsens over these steps

### 4.4.3 Line Search

Line search directly uses the optimum step size in order to provide the best improvement. Useful in specialized variations of gradient descent (but rarely used in vanilla GD bc computationally expensive). Armijo rule can be used in vanilla GD bc of efficiency

Let $J(\bar{w})$ be the obj fn being optimized and $\bar{g}_{t} $ be the descent direction at the beginning of the *t*th step with parameter vector $\bar{w}_{t} $. In `steepest-descent` method, the direction:

$\bar{g}_{t} = -\nabla J(\bar{w}_{t}) $

Although advanced methods might use other descent directions.

For below, assume $\bar{g}_{t} $ is not the steepest-descent direction (in order to preserve generalizty of the exposition).

From above:

$\bar{w} \Longleftarrow \bar{w} - \alpha_{t} \nabla J $

Clearly the parameter vector is updated as:

$\bar{w}_{t+1} \Longleftarrow \bar{w}_{t} + \alpha_{t}\bar{g}_{t} $

In line search, the lr \alpha_{t} is chosen in each step, so as to minimize the value of the obj fn at \bar{w}_{t+1}, where step size comptued as:

$\alpha_{t} = \argmin_{\alpha} J(\bar{w}_{t} + \alpha \bar{g}_{t}) $

After performing the next step, the gradient is computed at $\bar{w}_{t+1} $ for the next step. The gradient at $\bar{w}_{t+1} $ will be perpendicular to the search direction $\bar{g}_{t} $ or else $\alpha_{t} $ will not be optimal.

This result can be shown by observing that if the gradient of the obj fn at $\bar{w}_{t} + \alpha \bar{g}_{t} $ nas a **non-zer** dot product with the current movement direction $\bar{g}_{t} $, then one can improve the obj fn by moving an amount of either $+\delta $ or $-\delta $ along $\bar{g}_{t} $ from $\bar{w}_{t+1} $:

$J(\bar{w}_{t} + \alpha_{t}\bar{g}_{t} \pm \delta\bar{g}_{t}) \approx J(\bar{w}_{t} + \alpha_{t}\bar{g}_{t}) \pm \delta\bar{g}_{t}^{T}[\nabla J(\bar{w}_{t} + \alpha_{t} \bar{g}_{t})] $ where $\bar{g}_{t}^{T}[..] = 0 $ is the Taylor Expansion

#### Lemma 4.4.1

The gradient at the optimal point of a `line search` is always orthogonal to the current search direction

read:

- optimal point when the gradient is equal to zero (for current search direction)
- so the only way the gradient can be non-zero is if its pointing in a different direction
- I know we can get the orthogonality from the dot product being 0 but geometrically im not sure how we get there lol

A natural question arises as to how the minimization of Eq. 4.11 is performed. An important property of line-search is that that the obj fn $H(\alpha) = J(\bar{w} + \alpha\bar{g}_{t}) $ when expressed in terms of \alpha is often a `unimodal function`

- `unimodal function`
  - function with simgle maximum point
  - monotonicity

Main reason for being unimodal function is that typical ML settings that use the line-search method use quadratic, convex approximations of the original obj fn on which the search is done. Examples: Newton method, conjugate gradient method

Steps:

- 1. Identify a range [0, a_max] in which to perform the search
  - can be performed efficiently by evaluating the obj fn value at gegometrically increasing values of a (increasing every time by a factor of 2)
- 2. It is possible to use a variety of methods to narrow the interval such as the `binary search method`, the `golden-section search method`, and the `Armijo rule`
  - `binary search method` and `golden-section search method` are exact methods
    - leverage the unimodality of the obj fn in terms of the step size a.
  - `Armijo rule` is inexact
    - works even when $H(\alpha) = J(\bar{w}_{t} + \alpha \bar{g}_{t}) $ is multimodal/non-convex
    - therefore, has broader use than exact line-search methods

#### 4.4.3.1 Binary Search

We start by initializing the binary search interval for \alpha to [a, b] = [0, alpha_max]. In binary search over [a, b], the interval is narrowed by evaluating the obj function at two closely spaced points near (a+b)/2

We evaluate the obj fn at (a+b)/2 and (a+b)/2 + \eps, where \eps is a numerically small value like 10^-6. In other words, we compute:

- H[(a+b)/2]
- H[(a+b)/2 + \eps]

This allows us to evaluate whether the fn is increasingn or decreasing at (a+b)/2 by determining which of the two evaluations is larger.

- if fn is increasing at (a+b)/2
  - narrow the interval to [a, (a+b)/2 + \eps]
- otherwise
  - narrow to [(a+b)/2, b]

Repeat until an interval is reached with the required level of accuracy

#### 4.4.3.2 Golden-Section Search

Like binary search, start by init [a, b] = [0, alpha_max] however process of narrowing the interval is different

Basic principle in golden-section search is:

- use the fact that if we pick any pair of middle samples m1, m2 for a in the interval [a, b]
  - where a < m1 < m2 < b
- at least one of the intervals [a, m1] and [m2, b] can be dropped
  - In some cases, an even larger interval like [a, m2] and [m1, b] can be dropped
  - this is because the min value for unimodal fn must always lie in an adjacent interval to the choice of alpha \in {a, m1, m2, b} that yields the min value of H(alpha)
    - when alpha = a yields min, exclude (m1, b]
    - when alpha = m1 yields min, exclude (m2, b]
    - when alpha = m2 yields min, exclude [a, m1)
    - when alpha = b yields min, exclude [a, m2)
    - set new bounds [a, b] for these
  - At the end of the process, we are left with an interval containing either 0 or 1 evaluated point
  - if we have an interval containing no evaluated point
    - first select random point \alpha = p in the (reset) interval [a, b]
    - and then another point \alpha = q in the larger of the two intervals [a, p] and [p, b]
    - on the other hand, if left with an interval [a, b] containing a single evaluated point \alpha = p, then we select \alpha = q in the larger of the two intervals [a, p] and [p, b]
      - this yields another set of four points over which we can apply golden-section search
  - note this last bit with p and q I think there are some typos in the book

#### 4.4.3.3. Armijo Rule

The basic idea is that the descent direction $\bar{g}_{t}$ at the starting point $\bar{w}_{t} $ (i.e. at \alpha = 0) often deteriorates in thers of rate of improvement of obj fn as one moves further along this direction. The rate of improvement of the obj fn along the search direction at the starting point is: $|\bar{g}_{t}^{T}[\nabla F(\bar{w}_{t})]| $

Therefore, the (typical) improvement of the obj fn at a particular value of alpha can optimistically be expected to be $\alpha |\bar{g}_{t}^{T}[\nabla F(\bar{w}_{t})]| $ for most real-world objective fns

The `Armijo rule` is satisfied with a fraction $\mu \in (0, 0.5) $ of this improvement. A typical valyu of \mu is ~0.25 . In other words, we want to find the largest step-size \alpha satisfying the following:

$F(\bar{w}_{t}) - F(\bar{w}_{t} + \alpha \bar{g}_{t}) \geq \mu\alpha|\bar{g}_{t}^{T}[\nabla F(\bar{w}_{t})] | $

Note that for small enough values of \alpha, the condition above will always be satisfied. One can show using the finite-difference approximation that for infintesimally small values of alpha, the condition is satisfied at mu = 1. However, we want larget step size to ensure faster progress

What is the largest step-size one can use?

We test successively decreasing values of \alpha for the condition above, and stop the first time above the is satisfied. In backtracking line search, we start by testing $H(\alpha_{max}), H(\beta\alpha_{max})...H(\beta^{r}\alpha_{max})$ until the condition above is satisfied. At that point we use $\alpha = \beta^{r}\alpha_{max} $. here, \beta is a parameter drawn from (0, 1) and a typical value is 0.5

#### When to Use Line Search

Although line-search method can be shown to converge to at least a local optimum, it is expensive

When exact line search is required, the n steps is often relatively small, and the fewer the number of steps more than compensate for the expensive nature of individual steps

An important point with the use of line search is that convergence is guaranteed, even if the resulting solution is a local optimum

### 4.4.4 Initialization

GD starts at an initial point, and successively improves paramter vector at particular learning rate

Heurisitic initializatinos are good, esp for NNs because there are dependencies between optimization parameters
