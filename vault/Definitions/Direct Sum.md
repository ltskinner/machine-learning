
Builds on [[Sum of Subspaces]]


## Direct Sum

Suppose $V_1, ..., V_m$ are subspaces of V

The sum $V_1 + ... + V_m$ is a `direct sum` if each element of $V_1 + ... + V_m$ can be written in only one way as a sum $v_1 + ... + v_m$ where each $v_K \in V_k$

The direct sum is written as $V_1 \oplus ... \oplus V_m$

Laymans: each [[Subspace]] contributes unique "new" information

If you had 3 subspaces $V_1, V_2, V_3$ - every point can only be written as a unique combination of these subspaces. For instance, if you could write a point $p_1$ in terms of $V_1 + V_2$ **as well as** in terms of $V_2 + V_3$ then it is not a direct sum

### Condition for Direct Sum

Suppose $V_1, ..., V_m$ are subspaces of V.
Then $V_1, ..., V_m$ is a `direct sum` IFF the only way to write 0 as a sum $v_1 + ... + v_m$, where each $v_k \in V_k$ is by taking each $v_k$ equal to 0

See [[Direct Sum Condition]]

### Direct Sum of Two Subspaces

Suppose $U$ and $W$ are subspaces of $V$. Then
$U + W$ is a direct sum $\iff U \cap W = \{0\}$


## Proof Templates

### Direct Sum Condition

- For the direct sum **condition** give the proof symbolic example
	- [$]v, u \in V_1 + ... + V_m[/$] 
	- [$]v = v_1 + ... + v_m[/$] for [$]v_1 \in V_1, ..., v_m \in V_m[/$]  
	- [$]u = u_1 + ... + u_m[/$] for "...", so  
	- [$]0 = (v_1 - u_1) + ... + (v_m - u_m)[/$]
### Quantifier template to prove [$]A \subseteq B[/$]

1. [$]\forall x, (x \in A \implies x \in B)[/$]  
2. Expanded:  
[$]\forall x, (\exists u \in U, \exists w \in W : x = u + w) \implies (\exists w \in W, \exists u \in U : x = w + u)[/$]



## Flash Cards


- Direct Sum [$]\oplus[/$]
	- [$]V_1 \oplus ... \oplus V_m[/$]
- Definition of **direct sum** [$]\oplus[/$]
	- The sum [$]V_1 + ... + V_m[/$] is called a **direct sum** if each element of [$]V_1 + ... + V_m[/$] can be written **in only one** way as a sum [$]v_1 + ... + v_m[/$], where each [$]v_k \in V_k[/$]
- What is the formal **condition** for **direct sum**?
	- [$]V_1 + ... + V_m[/$] is a [$]\oplus[/$] IFF: 
	- [$]v_1 + ... + v_m = 0[/$] 
	- **only** when each [$]v_k \in V_k[/$] is [$]= 0[/$] 
	- This is stepping stone to **linear combination** condition
- What two conditions does the direct sum [$]\oplus[/$] capture?
	- 1. [$]V = U + W[/$]  
	- 2. [$]U \cap W = \{0\}[/$]
- What **condition** of direct sum prevents **bases** from _cancelling_?
	- [$]U \cap W = \{0\}[/$]
- Formal definition for direct sum of **two** subspaces around [$]\{0\}[/$]
	- [$]U + W[/$] is a direct sum [$]\iff U \cap W = \{0\}[/$]  (the only shared element is the zero vector/origin/additive identity)



- What is the **set analogy** for **direct sums** of subspaces?
	- Disjoint unions of subsets (union, but we know which element came from which set)
- In a **direct sum**, what does each subspace contribute?
	- Unique, independent directions !
- What more advanced LA concepts does the **direct sum** drive?
	- Linear combinations  
	- Basis construction  
	- Orthogonal decompositions  
	- Eigenspace decomposition  
	- PCA

- Suppose [$]U, V_1, V_2[/$] are subspaces of U and:  [$]V = V_1 \oplus U[/$] and [$]V = V_2 \oplus U[/$]
	- What are [$]V_1[/$] and [$]V_2[/$] called in relation to U, to make up U?  
	- Are [$]V_1[/$] and [$]V_2[/$] the same?
		- [$]V_1[/$] and [$]V_2[/$] are **complements** of U in V 
		- While they _can_ be identical, they are **not** required to be the same **unique** space
- What can we say about every subspace of V, direct sums, and the full vector space V?
	- Every subspace of V is part of a direct sum equal to V 
	- Suppose V is FD - then subspace W of V exists such that: [$]V = U \oplus W[/$]