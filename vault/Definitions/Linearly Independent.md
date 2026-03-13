Definition:

A list $v_1, ..., v_m$ if vectors in $V$ is called `linearly independent` if the only choice of $a_1, ..., a_m \in \mathbb{F}$ that makes
$a_1 v_1 + ... + a_m v_m = 0$
is $a_1 = ... = a_m = 0$

The empty list $()$ is also declared to be `linearly independent`

## Linearly DEPendent

A list of vectors in V is called `linearly dependent` if it is **NOT** `linearly independent`

In other words, a list $v_1, ..., v_m$ of vectors in V is linearly DEPendent if there exists $a_1, ..., a_m in \mathbb{F}$, not all 0, such that $a_1 v_1 + ... + a_m v_m = 0$

### Linear Dependence Lemma

Suppose $v_1, ..., v_m$ is a linearly dependent list in V
There exists $k \in \{1, 2, ..., m\}$ such that
$v_k \in \operatorname{span}(v_1, ..., v_{k-1})$

Furthermore, if $k$ satisfies the condition above and the kth term is removed from $v_1, .., v_m$, then the span of the remaining list equals $\operatorname{span}(v_1, ..., v_m)$

## Length of Linearly Independent List

In a finite-dimensional vector space, the length of *every* linearly independent list of vector is less than or equal to the length of every spanning list of vectors.

### Finite-dimensional Subspaces

Every [[Subspace]] of a [[Finite-Dimensional Vector Space]] is `finite dimensional`

