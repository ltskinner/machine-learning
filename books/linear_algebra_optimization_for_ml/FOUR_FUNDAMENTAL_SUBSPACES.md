# The Four Fundamental Subspaces of Linear Algebra

- row space
- column space
- right null space
- left null space

The core of studying matrices is to study linear transformations between vector spaces. These can be realized as matrix multiplication on the left of columns, or right of row, vectors

For column vector x: the image of the linear transformation Ax must span the columns of A

For row vector x: the image of the linear transformation xA must span the rows of A

### The Column Space

CASP = Column A SPace Right

If we have a matrix:

$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & 1 & 5 & 6 \\
0 & 0 & 0 & 8
\end{bmatrix}
$

The C(A) `column space` is columns {1, 2, 4}, which span A

The N(A)`(right) null space` is columns {3}, basically meaning x_3 can take any value

## The Row Space

RATS (Row = A^T Space) Left

Take A^T this time, and follow the same process

The C(A^T)`row space`, which span A

The N(A^T) `(left) null space`

## Ranks

- the column space and row space have equal dimension r = rank
- the left null space N(A) has dimension n - r (right null space)
- the left null space N(A^T) has dimension m - r

## General Notes

- The rank identifies the "core information" of A, where the row space and column space hold the meaningful parts of the information
- The left null space and right null space can be thought of as "slag" or redundancy, because they dont contribute to the effective rank
- The vectors spanning the row space are linearly independent
- The vectors spanning the column space are linearly independent

It seems the point of null spaces is to ensure completenss as we describe the behavior of A in operation.

## Core vs Null Space

- The rank captures the "core" or effective transformation part of A
- The null spaces encode the "failure modes" of A
  - The right null space shows where A loses information in the input
  - The left null space shows where A fails to span the output

### Right Null Space - Constraints on Inputs

The right null space (CASP right) consists of all input vectors x that are mapped to the zero vector by A

Aka: for Ax, if x \in N(A), it means x has no effect when multiplied by A - its "invisible" to A

These vectors lie in the part of R^n that A "completely annihilates" lmao

### Left Null Space - Constrains on Outputs

The left null space (RATS left) consists of all vectors y that are `othogonal` to the row space of A

This tells us which output directions in R^d cannot be reached by any input vector x

These are directions in the output space R^d that A fails to span

Note, if all output directions are spanned by the rows of A, then the left null space N(A^T) is trivial

## Bigger Picture

What we are describing is the process by which A maps R^n into R^d:

A = nxd matrix

A maps from R^d (input space) to R^n (the output space), making A act as a linear transformation

- A transforms R^d (input space) into R^n (output space):
  - that checks out Ax -> x is a column vector of n dimensions
- The `column space` identifies "reachable outputs in R^n"
  - is subspace of R^n (output space)
- The `(right) null space` identifies the "lost inputs" in R^d
  - is subspace of R^d (input space)
- The `row space` identifies the constraints on R^d
  - is subspace of R^d (input space)
  - the combinations of input vectors that are mapped into the column space
- The `(left) null space` identifies the "unreachable outputs" in R^n
  - is subspace of R^n (output space)
  - identifies outputs in R^n that are orthogonal to the column space (lost outputs)
  - which directions cannot be reached by input x: directions in output that R^n fails to span

As such:

- Column Space + Left Null Space (outers) = all of R^n
  - Every vector in R^n can be expressed as the sum of a vector in C(A) and one in N(A^T)
- Row space + Null Space (inners) = all of R^d
  - Every vector in R^d can be expressed as the sum of a vector in C(A^T) and one in N(A)

Ax = (nxd)(dx1) = nx1

- n is the output space
  - "this is where the result Ax lives"
- d is the input space
  - d = rank(A) + nullity(A)
  - d = rank(A) + N(A)
