
Definition:

The `dimension` of a finite-dimensional vector space is the length of any basis in the vector space


### 2.34 Basis length does not depend on basis

Any two bases of a finite-dimensional vector space have the same length

[[Basis]]

### 2.37 Dimension of a subspace

If $V$ is finite-dimensional and $U$ is a subspace of $V$, then $\operatorname{dim} U \leq \operatorname{dim} V$

### 2.38 Linearly independent lists of the right length is a basis

Suppose $V$ is finite-dimensional.
Then every `linearly independent` list of vectors in V of length $\operatorname{dim} V$ is a `basis` of $V$

[[Linearly Independent]]
### 2.39 Subspace of full dimension equals the whole space

Suppose that $V$ is finite-dimensional and $U$ is a subspace of $V$ such that $\operatorname{dim} U = \operatorname{dim} V$
Then $U = V$

### 2.42 Spanning list of the right length is a basis

Suppose $V$ is finite-dimensional.
Then every `spanning` list of vectors in $V$ of length $\operatorname{dim} V$ is a basis of $V$

[[Span]]

### 2.43 Dimension of a sum

If $V_1$ and $V_2$ are subspaces of a finite-dimensional vector space, then
$\operatorname{dim}(V_1 + V_2) = \operatorname{dim} V_1 + \operatorname{dim} V_2 - \operatorname{dim} (V_1 \cap V_2)$



## Flashcards


- Definition of **dimension** [$]\operatorname{dim} V[/$]
	- The length of any basis in the vector space  
	- (of a finite-dimensional vector space)
- Every **spanning** list of the right length is a ____?
	- Every **spanning** list of vectors in V of length [$]\operatorname{dim} V[/$] is a **basis** of [$]V[/$]
- Every **linearly independent** list of the right length is a ____?
	- Every **linearly independent** list of vectors in V of **length** [$]\operatorname{dim} V[/$] is a **basis** of V


- What is the dimension of subspace U of V?
	- [$]\operatorname{dim} U \leq \operatorname{dim} V[/$]
	- (If V is finite-dimensional and U is subspace of V)
- - Subspace U of full dimension equals the whole space of V means what between U and V?
	- If [$]\operatorname{dim} U = \operatorname{dim} V[/$]
	- Then [$]U = V[/$]
- For all subspaces of R^2 what possible **dimensions** can U have?
	- [$]\operatorname{dim}(U) \in \{0, 1, 2\}[/$]
- - If [$]\operatorname{dim}(U) = 0[/$] what is U = ?
	- [$]U = \{0\}[/$]
-  What is [$]\operatorname{dim}\{0\}[/$]
	- [$]\operatorname{dim}\{0\} = 0[/$]
- - What is the **dimension** of the sum of subspaces [$]V_1 + V_2[/$]? (regular sum)
	- [$]\operatorname{dim}(V_1 + V_2) = \operatorname{dim} V_1 + \operatorname{dim} V_2 - \operatorname{dim} (V_1 \cap V_2)[/$]
- - What **condition** in terms of **dimensions** makes a sum a **direct sum**?
	- [$]V_1 + ... + V_m[/$] is a direct sum [$]\iff[/$] 
	- [$]\operatorname{dim}(V_1 + ... + V_m) = \operatorname{dim} V_1 + ... + \operatorname{dim} V_m[/$]
- What is the **dimension** of the **direct sum** of two subspaces [$]V_1 \oplus V_2[/$] (almost direct sum, but not quite)
	- [$]\operatorname{dim} (V_1 + V_2) = \operatorname{dim} V_1 + \operatorname{dim} V_2[/$]  
	- [$]\iff V_1 \cap V_2 = \{0\}[/$]
- For subspace [$]U = \{(x, y, x+y, x-y, 2x) \in F^n : x, y, ... \in F \}[/$] - to produce a direct sum, what is the "dimensionality" of the complement subspace W?
	- Consider [$]... : x, y, ... \in F[/$] then dim(U) = 2 in this case, so the target needs to be [$]\operatorname{dim})(U \oplus W) = n = 5 = 2 + 3[/$] meaning dim(W) needs to be 3 so W needs to be of the form: 
	- [$]W = \{(...) \in F^5 : i, j, k \in F \}[/$]
- Consider [$]T(x,y) (x, y, x+y, x-y, 2x)[/$] - express this as a linear **map** (function) in terms of the dim F^? to F^?
	- [$]T: F^2 \rightarrow F^5[/$]
	- This works because e3, e4, e5 are all defined in terms of e1 and e2
- -Suppose we have a linear map [$]T: F^2 \rightarrow F^5[/$] - how do we denote the image of U?
	- Subspace U is the image of the linear map: 
	- [$]U = \operatorname{Im}(T)[/$]