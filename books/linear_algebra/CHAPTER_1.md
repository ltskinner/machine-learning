# Vectors in R^n and C^n, Spatial Vectors

- R = real numbers
- C = complex numbers

Addition = parallelogram

Multiplication = combining magnitudes in same direction

Dot product:

- a1*b1 + a2*b2 = scalar
- if dot product = 0, then orthogonal (perpendicular)

Norm (length) of vector:

- $\|u\| = \sqrt(u * u) = \sqrt(a_{1}^2 + a_{2}^{2} + ... + a_{n}^{2}) $

Unit length if $\|u\| = 1 = \frac{u}{\|u\|} $

Distance between vectors:

$d(u, v) = \|u - v\| = \sqrt{(a_{1 - b_{1}})^2 + (a_{2} - b_{2})^2 + ...} $

Angle $\theta $ between vectors:

$\cos{\theta} = \frac{u \cdot v}{\|u\| \|v\|} $

Projection of vector u onto v is:

$proj(v, v) = \frac{u \cdot v}{\|v\|^^2} u = \frac{u \cdot v}{v \cdor v} v $

basically the magnitude of u in the same direction which v points

Located Vector:

Hyperplane:

Set of points that satisfy the equation

$a_{1}x_{1} + a_{2}x_{2} + ... = b $ where a are coefficients living in vector b

Vectors in R^3 (Spatial Vectors), ijk notation:

- i = [1, 0, 0] unit vector in x direction
- j = [0, 1, 0] unit vector in y direction
- k = [0, 0, 1] unit vector in z direction

Any vector can be expressed in the form:

u = [a, b, c] = ai + bj + ck

They are all mutually orthogonal, so:

- i*i = 1
- j*j = 1
- k*k = 1

and

- i * j = 0
- i * k = 0
- j * k = 0

cross product:

- ad - bc

and

- i x j = k
- j x i = -k
- j x k = i
- k x j = -i
- k x i = j
- k x i = -j

Complex numbers:

- $i^2 = ii = -1 $
- $i = \sqrt(-1) $

z = (a, b) = (a, 0) + (0, b) = a + bi

Complex conjugate, Absolute Value:

- $z = \bar{a + bi} = a - bi $
- $z\bar{z} = (a + bi)(a - bi) = a^2 - b^2 i^2 = a^2 + (-1)(-b^2) $

Dot (Inner) Product in C^n:

$\|u\| = \sqrt(u \cdot u) = \sqrt{|z_{1}|^2 + |z_{2}|^2 + ...} $
