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

Taylor expansion can provide some insight:

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
