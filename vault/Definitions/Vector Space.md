
## Definition

A `vector space` is:
- a [[Set]] V, with
- addition on V
- scalar multiplication on V

Such that the following properties hold.

**Commutativity:**

$u + v = v + u$ for all $u, v \in V$

**Associativity:**

$(u + v) + w = u + (u + w)$, and

$(ab)v = a(bv)$ for all $u, v, w \in V$ and for all $a, b \in F$

**Additive Identity:**

There exists an element $0 \in V$ such that $b + 0 = v$ for all $v \in V$

**Additive Inverse:**

For every $v \in V$, there exists $w \in V$ such that $v + w = 0$

**Multiplicative Identity:**

$1v = v$ for all $v \in V$

**Distributive Properties:**

$a(u + v) = au + av$ and $(a + b)v = ab + bv$ for all $a, b \in F$ and all $u, v \in V$


## Flash Cards

### General

- Define a **vector space** (what is it, what properties does it have)?
	- A **set** V with **addition** and **scalar multiplication** on V, with:  
	- Associativity  
	- Commutativity  
	- Identity (Additive + Multiplicative)  
	- Inverse (Additive)  
	- Distributative
- What is a **real vector space**?
	- A vector space over [$]\mathbb{R}[/$]
- What do we call a **vector space** over [$]\mathbb{C}[/$]
	- A **complex vector space**
- Why are **fields** required for **vector spaces**?
	- Scalars must support addition, multiplication, and division to define linear scaling

### Property Focused


- Distributative
	- Define vector space **distributative properties** (formal) (scalar multiplication over addition)
	- [$]a(u + v) = au + av[/$], and  [$](a + b)v = av + bv[/$]  for all [$]a, b \in F[/$] and all [$]u, v \in V[/$]
- Commutativity of addition in [$]\mathbb{F}^n[/$] (formal)
	- If [$]x, y \in \mathbb{F}^n[/$], then [$]x + y = y + x[/$]
- - Define vector space multiplicative identity (formal)
	- [$]1v = v[/$] for all [$]v \in V[/$]
- Define vector space additive identity (formal)
	- There exists an element [$]0 \in V[/$] such that [$]v + 0 = v[/$] for all [$]v \in V[/$]
- - Scalar multiplication in [$]\mathbb{F}^n[/$] (formal). What is the result of scalar multiplication called?
	- The product of a number [$]\lambda[/$] and a vector in [$]\mathbb{F}^n[/$] is compared by multiplying each coordinate of the vector by [$]\lambda[/$]:  
	- [$]\lambda(x_1, ..., x_n) = (\lambda x_1, ..., \lambda x_n)[/$]  
	- Where [$]\lambda \in \mathbb{F}[/$] and [$](x_1, ..., x_n) \in \mathbb{F}^n[/$]
- Define addition on a vector space V?
	- A function that assigns an element [$]u + v \in V[/$] to each pair of elements [$]u, v \in V[/$] (think like binary operation of addition)
- Define vector space associativity (formal) (addition example and scalar multiplication example)
	- [$](u + v) + w = u + (v + w)[/$] and [$](ab)v = a(bv)[/$] for all [$]u, v, w \in V[/$] and for all [$]a, b \in F[/$]
- Addition in [$]\mathbb{F}^n[/$] (formal)
	- Defined by adding corresponding coordinates: 
	- [$](x_1, ..., x_n) + (y_1, ..., y_n) = (x_1 + y_1, ..., x_n + y_n)[/$]
- What is the critical caveat for scalar multiplication in subspaces?
	- When we say operations are closed under scalar multiplication:
	- [$]a \in \mathbb{F}[/$] and [$]u \in U[/$] implies [$]au \in U[/$]
	- This [$]\mathbb{F}[/$] actually means the field of the original vector space **V**
	- **NOT** the domain of the subspace
- Geometric model of the **additive inverse** of a 2d vector [$](x_1, x_2)[/$] (free)
	- Like just swap the head and the tail - the -x vector points in the opposite direction of the x vector, so naturally combining the two results in 0
- What is special about the **vector space** **additive identity**?
	- It is **unique** (one and only one additive identity)
- Define vector space **additive inverse** (formal)
	- For every [$]v \in V[/$], there exists [$]w \in V[/$] such that [$]v + w = 0[/$]
- What is special about **additive inverses** in **vector space**?
	- Every element in the vector space has a **unique** additive inverse
- What happens when scaling factors [$]\lambda[/$] are >1 or <1, what about when <0? (hybrid)
	- [$]\lambda[/$] < 1 shrinks the vector  
	- [$]\lambda[/$] > 1 stretches the vector  
	- [$]\lambda[/$] < 0 flips the vector to point in the opposite direction (similar to additive inverse)
- What is a number times the vector 0?
	- [$]a0 = 0[/$] for every [$]a \in F[/$]
- What is 0 times a vector? 0v = ?
	- $0v = 0$ for every $v \in V$
- What is the number -1 times a vector?
	- $(-1)v = -v$ for every $v \in V$
- Notation: what does -v mean? (simple dont overthink)
	- Additive inverse of v
- Additive inverse in [$]\mathbb{F}^n[/$], denoted as [$]-x[/$] (formal)
	- For [$]x \in \mathbb{F}^n[/$], the **additive inverse** of [$]x[/$], denoted by [$]-x[/$], is the vector [$]-x \in \mathbb{F}^n[/$] such that:  
	- [$]x + (-x) = 0[/$]  
	- Thus if [$]x = (x_1, ..., x_n)[/$], then [$]-x = (-x_1, ..., -x_n)[/$]
- What is a **scalar multiplication** on a set V?
	- A function that assigns an element $\lambda v \in V$ to each $\lambda \in \mathbb{F}$ and each $v \in V$
- Define vector space additive commutativity (formal)
	- $u + v = v + u$ for all $u, v \in V$
- Define complexification (formal set notation - simple)
	- [$]V_{\mathbb{C}} = \{u + vi : u, v, \in V \}[/$]
