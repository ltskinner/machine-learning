(of Subspaces)

## Sum of Subspaces

Suppose $V_1, .., V_m$ are subspaces of V

The `sum` of $V_1, .., V_m$ is the set of all possible sums of elements of $V_1, .., V_m$

$V_1 + ... + V_m = \{v_1 + ... + v_m : v_1 \in V_1, ..., v_m \in V_m \}$

Further, $V_1 + ... + V_m$ is the smallest subspace of V containing $V_1, ..., V_m$

## Direct Sum

Suppose $V_1, ..., V_m$ are subspaces of V

The sum $V_1 + ... + V_m$ is a `direct sum` if each element of $V_1 + ... + V_m$ can be written in only one way as a sum $v_1 + ... + v_m$ where each $v_K \in V_k$

The direct sum is written as $V_1 \oplus ... \oplus V_m$

Laymans: each [[Subspace]] contributes unique "new" information

If you had 3 subspaces $V_1, V_2, V_3$ - every point can only be written as a unique combination of these subspaces. For instance, if you could write a point $p_1$ in terms of $V_1 + V_2$ **as well as** in terms of $V_2 + V_3$ then it is not a direct sum

### Condition for Direct Sum

Suppose $V_1, ..., V_m$ are subspaces of V.
Then $V_1, ..., V_m$ is a `direct sum` IFF the only way to write 0 as a sum $v_1 + ... + v_m$, where each $v_k \in V_k$ is by taking each $v_k$ equal to 0

See [[Direct Sum Condition]]]

### Direct Sum of Two Subspaces

Suppose $U$ and $W$ are subspaces of $V$. Then
$U + W$ is a direct sum $\iff U \cap W = \{0\}$

