## Notation

The set of linear maps from $V$ to $W$

$\mathcal{L}(V, W)$

The set of linear maps from $V$ to $V$

- $\mathcal{L}(V, V)$
- $\mathcal{L}(V)$

Two equivalent notations for linear maps:

- $w = T(v)$
- $w = Tv$

Two equivalent notations:

$T(a_1 v_1 + ... + a_n v_n) = a_1 T(v_1) + ... + a_n T(v_n)$

Which allows us to avoid instantiating $a_j w_j$ variables 

## Definition

A `linear map` from $V$ to $W$ is a [[function]] $T: V \rightarrow W$ with the following properties.

`additivity`
$T(u + v) = Tu + Tv$ for all $u, v \in V$

`homogeneity`

$T(\lambda v) = \lambda(Tv)$ for all $\lambda \in \mathbb{F}$ and all $v \in V$

When $V$ and $W$ are [[Finite-Dimensional Vector Space]], this guarantees that:
- Both $V$ and $W$ have a [[basis]], which [[span]]s, and is [[linearly independent]]
- So when $a_1 v_1 + ... + a_n v_n = 0$ then $a_1 = ... = a_n = 0$

Linear maps take 0 to 0
- If $T$ is a linear map from $V$ to $W$ 
- Then $T(0) = 0$ 

### Linear map lemma

Suppose $v_1, ..., v_n$ is a basis of $V$ and $w_1, ..., w_n \in W$  
There exists a _unique_ **linear map** $T: \rightarrow W$ such that  
- $T v_k = w_k$  
- For each $k = 1, ..., n$

### Other Terms

`linear map` = `linear transform`


See: [[Fundamental Theorem of Linear Maps]]

### Operations

#### Additive Identity of $\mathcal{L}(V, W)$

- $0 \in \mathcal{L}(V, W)$ is defined by $0v = 0$

#### Addition on $\mathcal{L}(V, W)$

The **sum** $S + T$ is a **linear map**:
- $(S + T)(v) = Sv + Tv$ 

#### Scalar Multiplication on $\mathcal{L}(V, W)$

The **product** $\lambda T$ is a **linear map**: 
- $(\lambda T)(v) = \lambda (Tv)$

These properties make $\mathcal{L}(V, W)$ a [[Vector Space]]

#### Product of Linear Maps

**Definition** of **product** _of_ **linear maps** $ST \in \mathcal{L}(U, W)$

If $T \in \mathcal{L}(U, V)$ and $S \in \mathcal{L}(V, W)$ then _product_ $ST \in \mathcal{L}(U, W)$ defined by:  $(ST)(u) = S(Tu)$

$ST \neq TS$, so product of linear maps are not `commutative`

##### Properties

- Associativity 
	- $(T_1 T_2) T_3 = T_1 (T_2 T_3)$
	- $T_3$ maps into domain of $T_2$, and $T_2$ maps into domain of $T_1$
- Identity
	- $T I = I T + T$
	- First $I$ operates on $V$, second $I$ operates on $W$
- Distributive Properties
	- $(S_1 + S_2)T = S_1 T + S_2 T$, and
	- $S(T_1 + T_2) = S T_1 + S T_2$


## Properties of $\mathcal{L}(V, W)$
### Null Space of $\mathcal{L}(V, W)$

The **null** is the subset of V consisting of vectors that T maps/annihilates to 0 
- $\operatorname{null} T = \{v \in V : Tv = 0 \}$
- Null T is a **subspace** of V

`null space` = `kernel`

#### Null Space Example

Given $T(x, y, z) = (x, y)$
- $\operatorname{null} T = \{(0, 0, z) \}$ 

See [[Injective]]

### Range of $\mathcal{L}(V, W)$

The **range** of T is the subset of W consisting of those vectors that are equal to $Tv$ for some $v \in V$
- $\operatorname{range}T = \{Tv : v \in V \}$, defined in terms of $V$
- Range T is a **subspace** of W

##### Range Example

Given $T(x, y, z) = (x, y)$:
- $\operatorname{range} T = \{(a, b, 0) \}$

See [[Surjective]]

#### Example of Null T = Range T

for $T \in L(R^4)$
- $T(x_1, x_2, x_3, x_4) = (x_3, x_4, 0, 0)$
- $\operatorname{null} T = \{(x_1, x_2, 0, 0) \}$  
- $\operatorname{range} T = \{(y_1, y_2, 0, 0) \}$  
- b/c $T(...) = (x_3, x_4, 0, 0) = (y_1, y_2, y_3, y_4)$


## Solutions

- What does a nonzero solution to SLE entail?
	- At least one solution, but really infinitely many  
	- Not linearly independent (not full rank)  
	- There is one or more free variables
- Homogeneous system of linear equations RE: solution space for variables < equations
	- A homogeneous SLE w/ # variables < # equations = has no solution _for some choice of the constant terms_
- Homogeneous system of linear equations RE: solution space for variables > equations
	- A homogeneous system of linear equations with: # num variables > # equations = has nonzero solutions
- What is a **nonzero solution** to a homogeneous system of linear equations ([$]Ax = 0[/$] but NOT + b)?
	- There is at least **one solution** vector x where not all variables are zero



## Proof Techniques
### Construct Linear Map

Construct $T : V \rightarrow W$ in such a way that T is **surjective** and is fully defined in terms of v1, …, vn as basis

- $V, W$ are finite-dim, giving them both **basis**  
- $\operatorname{dim} V = n$ and $\operatorname{dim} W = m$, $n \geq m$  
- $T(v_1) = w_1, ..., T(v_m) = w_m$ (remember $n \geq m$)  
- $T(v_{m+1}) = ... = T(v_n) = 0$ (remember $n \geq m$)  
- $w_1, ..., w_m \subseteq \operatorname{range} T$, which **spans** $W$ b/c **basis**

### When, why, and how do we employ **linear map lemma** in proofs?

When: we introduce/define a linear map like $T \in L(V, W)$
Why: to ensure $T$ is **well-defined** over the **entire space**  
How: If V has a basis, and W has a basis, then $T(v_j) = w_j$ so $T$ is a **well-defined** linear map on the **entire space**

### Core Proof Engines

What are the 4 core engines for LA proofs?

- 1. Basis / extend-a-basis  
- 2. Linear independence / spanning  
- 3. Rank-nullity  
- 4. Define a **map** on a **basis**, then extend linearly

### When Use

- When use **basis / extend-a-basis** engine
	- To **construct** a subspace, complement, direct sum, linear map w/ **specific behavior**
- When use **linear independence / spanning** engine
	- When proving something **is** a basis, or showing a map is injective/surjective
	- See: [[Linearly Independent]]
	- See: [[Span]]
- What are the (2) tells you can use rank-nullity to prove?
	- When given $T: F^n \rightarrow F^m$
	- When given the null space explicitly (like the conditions)
	- See: [[Fundamental Theorem of Linear Maps]]
- When use **definine map on basis, then extend linearly** engine
	- When proving:  
		- **existence** of a linear map  
		- **example** map w/ special properties  
		- inverse-like map  
		- left/right inverse

### How use **basis / extend-a-basis** engine

1. **Choose** a basis for the known subspace/object  
2. **Extend** to basis of the **whole space**  
3. Use the added basis vectors to define the **missing piece**

### How use **define map on basis, then extend linearly** engine

1. Pick a basis $v_1, ..., v_n$ of **domain**  
2. Choose target vectors $w_1, ..., w_n$ in the **codomain**  
3. Define $T(v_i) = w_i$
4. Extend linearly $T(a_1 v_1 + ... + a_n v_n) = a_1 w_1 + ... + a_n v_n$


### Verbiage for defining linear map T in a proof

"Define $T: V \rightarrow W$ on the basis $v_1, ..., v_n$  
...  
By the linear map lemma, this defines a linear map $T \in L(V, W)$"

### Prove Linearity

Workflow of using additive condition to prove linearity of linear map

$T(p + q) = T(p) + T(q)$ - if $\neq 0$ then not linear  
$T(p + q) - (T(p) + T(q)) = 0$ - if $\neq 0$ then not linear

### Not Subspace

How to prove that linear map T is not a subspace of L(V, W)?

Find two maps T1 and T2 that meet the conditions whose sum is not in L(V, W)


### S = TE

Suppose $\exists E \in L(W, W)$ such that $S = TE$, what does this tell us about S and T?
$\operatorname{range} S \subseteq \operatorname{range} T$

### T = ES

Suppose $\exists E \in L(W, W)$ such that $T = ES$, what does this tell us about S and T?
$\operatorname{null} S \subseteq \operatorname{null} T$




## Flash Cards

- Definition of **linear map** (include properties)
	- A linear map from [$]V[/$] to [$]W[/$] is a function [$]T: V \rightarrow W[/$] with
		- **Additivity**
			- [$]T(u + v) Tu + Tv[/$] for all [$]u,v \in V[/$]
		- **Homogeneity**
			- [$]T(\lambda v) = \lambda (Tv)[/$] for all [$]\lambda \in \mathbb{F}[/$] and [$]v \in V[/$]
- How do we denote the set of **linear maps** from [$]V[/$] to [$]V[/$]?
	- [$]\mathcal{L}(V)[/$] or [$]\mathcal{L}(V) = \mathcal{L}(V, V)[/$]
- What are the two notations for **linear maps**?
	- [$]Tv[/$] and [$]T(v)[/$]
- What does [$]\mathcal{L}(V, W)[/$] denote?
	- The set of **linear maps** from [$]V[/$] to [$]W[/$]
- What is [$]T(a_1 v_1 + ... + a_n v_n)[/$] directly equal to? Why would we write the other form?
	- [$]T(a_1 v_1 + ... + a_n v_n) = a_1 T(v_1) + ... + a_n T(v_n)[/$]We write it like this to avoid instantiating any [$]a_j w_j[/$] variables
- What is another term for **linear transform**?
	- **Linear map**
- What distinguishes [$]T : V \rightarrow W[/$] and [$]T \in (V, W)[/$] theoretically and how do we **convert** the former into the latter?
	- [$]T : V \rightarrow W[/$] is just an abstract function
	- [$]T \in (V, W)[/$] is a **linear map** - more tightly constrained properties
	- Convert through [$]T(a_1 v_1 + ... + a_x v_x) = a_1 w_1 + ... + a_x w_x[/$]
	- If condition of **V** is known, use basis [$]v_1, ..., v_n[/$] (injectivity)
	- If condition of **W** is known, use basis [$]w_1, ..., w_m[/$] (surjectivity)
- What distinguishes a linear function L(v) from an affine function?
	- $L(u + v) = L(u) + L(u)$
	- $L(\lambda v) = \lambda L(v)$  
	- NO constant term  
	- Maps origin to origin (0, ..., 0)  
	- Ex. $L(x) = Ax$
- What distinguishes an **affine** function?
	- Preserves linearity but includes a shift (change of basis)  
	- Preserves lines and parallelism  
	- [$]f(x) = Ax + b[/$]  
	- Does NOT preserve origin
- - When is [$]f(x) = mx + b[/$] a linear map
	- IFF [$]b = 0[/$], making it **homogeneous**
- - What does the _identity operator_ matrix [$]I[/$] do as a **linear map**?
	- [$]I[/$] is a **linear map** on some **vector space** that takes each element to itself
- - When given that a linear map [$]T \in L(V, W)[/$] that is **surjective**, what is our starting basis? How do we relate this to the other subspace?
	- Because **surjective**, let [$]w_1, ..., w_m[/$] be a **basis** of W. Then [$]\exists v_i \in V[/$] where [$]T(v_i) = w_i[/$] for [$]i = 1, ..., m[/$]


