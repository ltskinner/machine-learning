See [[Direct Sum]]

## Condition

Suppose $V_1, ..., V_m$ are subspaces of V.
Then $V_1, ..., V_m$ is a `direct sum` IFF the only way to write 0 as a sum $v_1 + ... + v_m$, where each $v_k \in V_k$ is by taking each $v_k$ equal to 0

- Requires $\iff$ both directions

First direction:
- If $V_1 + ... + V_m$ is a direct sum
	- the definition of direct sum implies that the only way to write 0 as a sum $v_1 + ... + v_m$, where each $v_k \in V_k$ is by taking each $v_k$ equal to 0

Second direction
- Suppose the only way to write 0 as a sum $v_1 + ... + v_m$, where each $v_k \in V_k$, is by taking each $v_k$ equal to 0.
	- To show that $V_1 + ... + V_m$ is a direct sum, let $v \in V_1 + ... + V_m$.
	- We can write:
		- $v = v_1 + ... + v_m$
	- For some $v_1 \in V_1, ..., v_m \in V_m$
	- To show this representation is unique, suppose we also have:
		- $v = u_1 + ... + u _m$ (yes $v = u_i$)
	- Where $u_1 \in V_1, ..., u_m \in V_m$
	- Subtracting these two equations we have
		- $0 = (v_1 - u_1) + ... + (v_m - u_m)$
	- Because $v_1 - u_1 \in V_1, ..., v_m - u_m \in V_m$, the equation above implies that each $v_k - u_k$ equals zero
- Thus, $v_1 = u_1, ..., v_m = u_m$ as desired


## Direct Sum of Two Subspaces

Suppose $U$ and $W$ are subspaces of $V$. Then
$U + W$ is a direct sum $\iff U \cap W = \{0\}$

- $\iff$ so need both directions

First:
- Suppose that $U + W$ is a direct sum
	- If $v \in U \cap W$, then $0 = v + (-v)$, where $v \in U$ and $-v \in W$
		- By the unique representation of 0 as the sum of a vector in $U$ and a vector in $W$, we have $v = 0$
		- Thus, $U \cap W = \{0\}$, completing the proof in one direction
Second:
- Suppose $U \cap W = \{0\}$
	- To prove that $U + W$ is a direct sum, suppose $u \in U, w \in W$ and $0 = u + w$
		- To complete the proof, we only need to show that $u = w = 0$
		- The equation above implies that $u = -w \in W$
		- Thus $u \in U \cap W$
		- Hence, $u = 0$, which by the equation above implies that $w = 0$, completing the proof