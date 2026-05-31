
## Definition of Fundamental Theorem of Linear Maps


Suppose V is **finite-dimensional** and $T \in L(V, W)$
Then, **range** T is finite-dimensional;
and $\operatorname{dim} V = \operatorname{dim} \operatorname{null} T + \operatorname{dim} \operatorname{range} T$

## Proof Templates

### dim null ST 2

Consider $T \in L(V, W)$ where **U** is a subspace of **W**  
What is $\{v \in V | Tv \in U\}$ a subspace of?  
What is the dim of its **null**?  
For $STv$ what is U subspace of? (2 other spaces)


$\{v \in V | Tv \in U\}$ is a subspace of **V**  
$\operatorname{dim}\operatorname{null} \{v \in V | Tv \in U\} = \operatorname{dim}\operatorname{null} T + \operatorname{dim}(U \cap \operatorname{range}T)$
  
$(ST)(v) = S(T(v)) = S(w)$ so $U \subseteq \operatorname{range} T \subseteq W$, which is the functional **domain** of S

### dim null ST

General proof of $\operatorname{dim}\operatorname{null} ST \leq \operatorname{dim} \operatorname{null} S + \operatorname{dim} \operatorname{null} T $ for $T \in L(V, U)$ and $S \in L(U, W)$

1. Show $\operatorname{null} ST = \{v \in V | Tv \in \operatorname{null} S \}$
2. And $\operatorname{dim} \{v \in V | Tv \in \operatorname{null} S \} = \operatorname{dim} \operatorname{null} T + \operatorname{dim}(\operatorname{null} S \cap \operatorname{range} T)$

### dim range ST

General proof of $\operatorname{dim}\operatorname{range} ST \leq \operatorname{min}\{\operatorname{dim} \operatorname{range} S, \operatorname{dim} \operatorname{range} T \}$ for $T \in L(V, U)$ and $S \in L(U, W)$

1. Show $\operatorname{dim} \operatorname{range} ST \leq \operatorname{dim} \operatorname{range} S$ (easy b/c S(Tv) filtered by Tv)  
2. Show $\operatorname{dim} \operatorname{range} ST \leq \operatorname{dim} \operatorname{range} T$  
(show $\operatorname{null} T \subseteq \operatorname{null} ST$ - pick arbitrary element in null T and show it also resides in null ST)


