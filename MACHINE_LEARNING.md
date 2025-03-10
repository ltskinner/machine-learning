# Machine Larning

Starting with the normal equation $A\bar{x} = \bar{b}  $

- $\bar{x} $ is the `scaling factors` corresponding to the columns of $A $
  - this does **not** live in either the row or the column space (it lives in R^d)
  - **HOWEVER** the useful components of x to help define b do live in the `row space` (and the lost inputs reside in the `(right) null space`)
- the columns of $A $ represent the independent directions that span some space
- $\bar{b} $ lives in the column space (if the system is consistent)

In many cases:

- $\bar{x} $ is some input data (features)
- $\bar{b} $ is some output data (labels or predictions)
- $A $ may not be known

When A is not known, we seek to learn the values of A which project the input data into the space where b resides by using A - we seek to learn A

The structure of A is partially informed by b, because we seek a consistent system where b resides in the column space of A

## Specific Scenarios

### Case 1: A is Square

- A has an equal number of rows and columns
- Each input x should have a direct, and unique correspondence to the output b
- Exactly one solution exists

### Case 2: A is Tall

- A has many more rows than columns
- over-determined system
- typically inconsistent (unless b lies exactly in column space)
- a best fit solution can be reached if the system is consistent by minimizing the error
- we have less information in the inputs x than is required for the output b
  - each bit of information in x corresponds to an "equation" in A
  - there are more "equations" aka "constraints" than inputs x

### Case 3: A is Wide

- A has many more columns than rows
- under-determined system
- there are more inputs x than "equations" aka "constraints"
- we have more information in the inputs x than is required for the output b
  - potentially infinite number of solutions, because many ways to reach b from x
  - here, we constrain the solutions via regularization
