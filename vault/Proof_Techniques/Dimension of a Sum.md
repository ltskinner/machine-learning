
2.43 Dimension of a sum

If V_1 and V_2 are subspaces of a finite-dimensional vector space, then
$\operatorname{dim} (V_1 + V_2) = \operatorname{dim} V_1 + \operatorname{dim} V_2 - \operatorname{dim} (V_1 \cap V_2)$

Proof:

- Let $v_1, ..., v_m$ be a basis of $V_1 \cap V_2$
	- Thus $\operatorname{dim} (V_1 \cap V_2) = m$
	- Because $v_1, ..., v_m$ is a `basis` of $V_1 \cap V_2$
		- It is `linearly independent `in $V_1$
		- Hence, this list can be extended to a basis of V_1 (by itself) as
			- $v_1, ..., v_m, u_1, ..., u_j$ of $V_1$
			- Thus, $\operatorname{dim} V_1 = m + j$
		- Also extend to a basis of V_2 (by itself) as
			- $v_1, ..., v_m, w_1, ..., w_k$ of $V_2$
			- Thus, $\operatorname{dim} V_2 = m + k$
		- We will show that
			- **2.44** $v_1, ..., v_m, u_1, ..., u_j, w_1, ..., w_k$
		- is a basis of $V_1 + V_2$
		- This will complete the proof because:
			- $\operatorname{dim}(V_1 + V_2) = m + j + k$
			- $= (m + j) + (m + k) - m$
			- $= \operatorname{dim} V_1 + \operatorname{dim} V_2 - \operatorname{dim} (V_1 \cap V_2)$
- The list **2.44** is contained in $V_1 \cup V_2$ and thus contained in $V_1 \cap V_2$
- The **span** of 2.44 contains $V_1$ and contains $V_2$ and hence is equal to $V_1 + V_2$
	- (LTS but is not a direct sum because overlapping stuff $v_1, ..., v_m$)
- Thus to show that **2.44** is a basis of $v_1 + V_2$ we only need to show that it is `linearly independent`
- To prove that **2.44** is linearly independent, suppose
	- $a_1 v_1 + ... + a_m v_m + b_1 u_1 + ... + b_j u _j + c_1 w_1 + ... + c_k w_k = 0$
	- where all $a$'s, $b$'s, and $c$'s are scalars
	- We need to prove that all a, b, c = 0
	- Rewrite above as
		- **2.45** $c_1 w_1 + ... + c_k w_k = -a_1 v_1 - ... - a_m v_m - b_1 u_1 - ... - b_j u_j$
		- Which shows that $c_1 w_1 + ... + c_k w_k \in V_1$
		- All the $w$'s are in $V_2$, so this implies that $c_1 w_1 + ... + c_k w_k \in V_1 \cap V_2$
		- Because $v_1, ..., v_m$ is a basis of $V_1 \cap V_2$, we have
			- $c_1 w_1 + ... + c_k w_k = d_1 v_1 + ... + d_m v_m$
			- for some scalars $d_1, ..., d_m$
		- But $v_1, ..., v_m, w_1, ..., w_k$ is linearly independent so the last equation implies that the $c$'s (and $d$'s) equal 0
			- (LTS use the Linear Independence condition to remove "unused" terms in 2.45)
		- Thus, **2.45** becomes the equation
			- $a_1 v_1 + ... + a_m v_m + b_1 u_1 + ... + b_j u_j = 0$
		- Because the list $v_1, ..., v_m, u_1, ..., u_j$ is linearly independent, this equation implies that all the $a$'s and $b$'a are 0, completing the proof
