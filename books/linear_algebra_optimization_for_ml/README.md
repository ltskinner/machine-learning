# Linear Algebra and Optimization for Machine Learning

[Linear Algebra and Optimization for Machine Learning](https://www.amazon.com/Linear-Algebra-Optimization-Machine-Learning/dp/3030403432)

This book goes out of its way to teach linear algebra and optimization with ML examples

The exercises within the text of the chapter should be solved as one reads the chapter in order to solidify the concepts.

The exercises at the end of the chapter are intended to be solved as refreshers after completing the chapter

Notations:

Vectors or multi-dimensional data points are $\bar{X} or \bar{y}  $ - the bar is the distinguishing feature

Vector dot products are denoted by centered dots, such as $\bar{X} \cdot \bar{Y}  $. A matrix is denoted in captial letters without a bar, like $R $.

The $n \times d $ matrix corresponding to the entire training dataset is denoted by D with n data points and d dimensions. The individual data points in D are therefore d-dimensional row vectors and are often denoted by $\bar{X}_{1} ... \bar{X}_{n}  $.

Conversely, vectors with one component for each data point are usually n-dimensional column vectors. An example is the n-dimensional column vector \bar{y} of class variables of n data points.

A observed value $y_{i}$ is distinguised from a predicted value $\hat{y_{i}}$

## [Chapter 1. Linear Algebra and Optimization: An Introduction](./CHAPTER_1.md)

## [Chapter 2. Linear Transformations and Linear Systems](./CHAPTER_2.md)

## [Chapter 3. Eigenvectors and Diagonalizable Matrices](./CHAPTER_3.md)

can the notion of `matrix similarity` be abused?

Like what if we have two similar matrices of different dimensions (if thats possible) - can we optimize on a smaller matrix for a less fine tuned representation, and then apply to the larger more fine grained matrix?

Or vice versa, can we optimize on the more fine tuned matrix then simplify to the smaller matrix to reduce weights size?

Update after re-doing: I think PCA does something like this based on the eigenvectors and eigenvalues, so occurs in the eigenspace instead of the native space

## [Chapter 4. Optimization Basics: A Machine Learning View](./CHAPTER_4.md)

## Cleanup

At end of book, pull togethter all the named Definitions and Observations
