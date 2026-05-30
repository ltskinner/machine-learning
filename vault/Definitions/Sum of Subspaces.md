
Builds on [[Subspace]]

## Definition

Suppose $V_1, ..., V_m$ are sub**spaces** of $V$

The **sum** is denoted by $V_1 + ... + V_m$, which is the **se**t of all possible sums of elements: $V_1 + ... + V_m = \{v_1 + ... v_m | v_1 \in V_1, ..., v_m \in V_m \}$

For simple $U + W$ we would write this as:
$U + W = \{u + w : u \in U, w \in W \}$

For a [[Vector Space]] defined as $V = V_1 + ... + V_m$, we express individual vectors $v$ as:
- $v \in V \equiv v \in V_1 + ... + V_m \equiv v = v_1 + ... + v_m$
- for some $v_1 \in V_1, ..., v_m \in V_m$

Further, $V_1 + ... + V_m$ is the smallest subspace of $V$ containing $V_1, ..., V_m$


## Example 1

- $U = \{(x, x, y, y) \in F^4 | x, y \in F \}$
- $W = \{(x, x, x, y) \in F^4 | x, y \in F \}$
	- $(a, a, b, b) + (c, c, c, d) = (a+c, a+c, b+c, b+d)$
		- which if we rename these as other symbols 
	- $(i, i, j, k) = (x, x, y, z)$, so 
	- $U + W = \{(x, x, y, z) \in F^4 | x, y, z \in F \}$

### Example 2


- $U = \{(x, -x, 2x) \in F^3 | x \in F \}$
- $W = \{(x, x, 2x) \in F^3 | x \in F \}$
	- $U + W = (a + b, -a + b, 2a + 2b)$
	- $U + W = (a + b, -a + b, 2(a + b))$
	- $U + W = ... (x, y, 2x) ...$
	- $U + W = \{(x, y, 2x) \in F^3 | x, y \in F \}$


## Flashcards

- Laymans alternative to: the **sum of subspaces** is the "smallest containing subspace"
	- The minimum set (closed under linear combinations) that includes all summands needed to reach the sums
	- **Critical**: it is "like" a union but NOT. Has added requirement of preserving linear combinations
- What size of subspace does the sum of subspaces produce?
	- The smallest subspace that contains all the summands (every subspace containing all the summands also contains the sum)
- Why do we work with the **sum** of subspaces? (as opposed to the union)
	- The sum of subspaces is a subspace
	- Whereas, the **union** of subspaces is **rarely** a subspace
- - What is the symbolic representation of the Union of two subspaces resulting in a subspace?
	- [$]U + W = U \iff W \subseteq U[/$] is a subspace of V
- - If U is a subspace of [[Vector Space]] V, what is U + U?
	- U + U = U, which is a subspace of V