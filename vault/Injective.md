
 The function $T: V \rightarrow W$ is **injective** if: 
- $Tu = Tv \implies u = v$
- $\operatorname{null} T = \{v \in V : Tv = 0 \} = \{0\}$
	- the null space is empty (only the zero vector)
- No inputs are annihilates - all inputs are utilized

If the basis of V, $v_1, ..., v_n$ is **linearly independent**, then T is injective:
$T(a_1 v_1, ..., a_n v_n) = 0$ so $a_1 = ... = a_n = 0$ which is definition of linear independence

- Linear maps to **lower**-dimensional spaces are _NOT_ **injective** 
	- If dim V > dim W, then no **linear map** from V to W is **injective** 

##### Injective Example

Given $T(v_1, v_2, v_3)$
- $T(a_1 v_1, a_2 v_2, a_3 v_3) = (a_1 w_1, a_2 v_2)$
	- NOT injective b/c $\operatorname{dim} \operatorname{null} T = 1$

Given $T(v_1) = 0$
- if $v_1 \neq 0$ then NOT **injective**