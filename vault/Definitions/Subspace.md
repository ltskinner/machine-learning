A subset U of V is called a subspace of V if:
- U is also a [[Vector Space]] with the same
	- Additive [[Identity Element]]
	- Addition
	- Scalar Multiplication

Formally:

**Additive identity**

$0 \in U$

**Closed under addition**

$u, w \in U \implies u + w \in U$

**Closed under scalar multiplication**
$a \in F, u\in U \implies au \in U$

## Examples

### Crazy-ass subspace definitions

- Give two examples of crazy-ass subspace definitions
	- $U = \mathbb{Z}^2 \subset \mathbb{R}^2$  
		- $\frac{1}{2} = a \in R$ the Vector Space  
		- $au = \frac{1}{2}(1, 0) = (\frac{1}{2}, 0) \notin U$
	- $U = \{(x, 0) | x \in R \} \cup \{(0, y) | y \in R \}$
		- $10 = a \in \mathbb{R}$ so $au = 10(1, 0) = (10, 0)$
		- $(1, 0) + (0, 1) = (1, 1) \notin U$


## Flash Cards

- What are the conditions for a subspace (formal)
	- Sub**set** [$]U[/$] of [$]V[/$] is a sub**space** of [$]V[/$] IFF [$]U[/$] satisfies:  
	- additive identity: [$]0 \in U[/$]  
	- **closed** under addition: [$]u, w \in U[/$] implies [$]u + w \in U[/$]  
	- **closed** under scalar multiplication: [$]a \in \mathbb{F}[/$] and [$]u \in U[/$] implies [$]au \in U[/$]
- What is the additive identity for subspace addition?
	- [$]\{ 0 \}[/$]
- Definition of **subspace** (formal)
	- A sub**set** [$]U[/$] of [$]V[/$] is called a subspace of [$]V[/$] if [$]U[/$] is also a **vector space**, with the same properties as [$]V[/$]:  
	- additive identity  
	- addition  
	- scalar multiplication
- - Which subspaces have additive inverses?
	- Literally only trivial ones like [$]\{ 0 \}[/$] have one
- What is a **non-homogeneous equation** and what does it tell you about **subspaces**?
	- Have constraints (bias terms)  
		- Ex. x + 2y = 3  
		- Tell you: do **NOT** reside in a subspace b/c doesnt pass thru origin (no 0 vector)
- What is a **homogeneous equation** and what does it tell you about **subspaces**?
	- Homogeneous equations have **no constraints** (no bias terms)  
		- Ex. x + 2y = 0  
		- Tell you: possibly reside in a **subspace**
- What is another name for a shifted set that "misses" the origin?
	- Its an **affine** set, as opposed to linear
- What is the general symbolic form set definition for **every** line through an origin?
	- [$]U = \operatorname{span}\{(a, b)\} = \{t(a, b) : t \in \mathbb{R} \}[/$] for some nonzero [$](a, b) \in \mathbb{R}^2[/$]
- Whats a simplified expression of [$]U = \operatorname{span}\{(a, b)\} = \{t(a, b) : t \in \mathbb{R} \}[/$]
	- [$]U = \operatorname{span}\{v\} = \{av : a \in \mathbb{R} \land v \in \mathbb{R}^2 \}[/$]
- What does the **Zero Vector Test** tell you about whether a set is a subspace?
	- A set can only be a subspace if it contains the zero vector (aka, the **additive identity**)
- Whats the geometric meaning of the Zero Vector Test?
	- The zero vector is the origin  
	- subspaces must pass through the origin (are not "shifted" by bias terms like + b)
- Under what operation makes **every** subset of V a subspace of V?
	- The **intersection** [$]\cap[/$] of every subspace
- Under what condition is the union of two subspaces of V also a subspace of V?
	- IFF one of the subspaces is contained in the other

