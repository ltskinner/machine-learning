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
