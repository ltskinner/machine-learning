
Definition:

A function $p : \mathbb{F} \rightarrow \mathbb{F}$ is called a `polynomial` with coefficients in $\mathbb{F}$ if there exists $a_0, ..., a_m \in \mathbb{F}$ such that
$p(z) = a_0 + a_1 z + a_2 z^2 + ... a_m z^m$
for all $z \in \mathbb{F}$

$\mathcal{P}(\mathbb{F})$ is the set of all polynomials with coefficients in $\mathbb{F}$



## Flashcards



- Definition of a **polynomial**
	- A function $p : F \rightarrow F$ is called a polynomial with coefficients in F if there exists $a_0, ..., a_m \in F$ such that: 
	- $p(z) = a_0 + a_1 z + a_2 z^2 + ... + a_m z^m$
	- for all $z \in F$
- What determines the coefficients of a polynomial?
	- The polynomial itself defines the unique coefficients (something to do with **linear combinations** likely)
- What does $\mathcal{P}(\mathbb{F})$ express?
	- $\mathcal{P}(\mathbb{F})$ is the set of all polynomials with coefficients in F
- What type of space is the set of all polynomials $\mathcal{P}(\mathbb{F})$ ?
	- $\mathcal{P}(\mathbb{F})$ is a **vector space** over F
- What is $\mathcal{P}(\mathbb{F})$ a **subspace** of?
	- $F^F$ the **vector space** of **functions** from F to F
- What does $\mathcal{P}_{m}(\mathbb{F})$ denote?
	- The set of all polynomials with coefficents $a \in F$ and a degree at most _m_
- What is the degree of a polynomial?
	- The number of coefficents $a_0 + a_1 a + ... a_m a^m$ where $a_m \neq 0$
	- Denoted as $\text{deg} p$
- What is the degree of a polynomial that is identically 0?
	- $\text{deg} p = - \infty$
- What does $\mathcal{P}_{m}(\mathbb{F}) =$
	- $\mathcal{P}_{m}(\mathbb{F}) = \text{span}(1, z, .., z^m)$
	- Making $\mathcal{P}_{m}(\mathbb{F})$ a **finite-dimensional** vector space
- What is $\operatorname{dim} \mathcal{P}_{m}(\mathbb{F})$
	- $\operatorname{dim} \mathcal{P}_{m}(\mathbb{F}) = m + 1$
	- b/c std basis $1, z, ..., z^m$ length is $m+1$
- What is the standard basis of $\mathcal{P}_{m}(\mathbb{F})$
	- The list $1, z, ..., z^m$
- Define a basis of a polynomial $\mathcal{P}_n(\mathbb{F})$ with a linear constraint $p(a) = 0$
	- Standard basis of $\mathcal{P}_n(\mathbb{F}) = \{1, z, z^2, ..., z^n\}$
	- w/ linear constraint $a$, basis is $(z - a)\{1, z, z^2, ..., z^{n-1}\}$
	- So basis is $\{(z - a), z(z - a), z^2(z - a), ..., z^{n-1}(z - a)\}$
	- **BUT** this is only a basis of $\mathcal{P}_{n-1}(\mathbb{F})$
- What do polynomials have that _may_ **reduce** the dimension of a subspace?
	- Polynomials may share a **common root** (or other constraint)
- Can $\mathcal{P}_3(\mathbb{F})$ be spanned by a list of polynomials that do not have degree 2?
	- Yes because $z^3$ may contain $z^2$ terms within it, however these are now linearly **dep**endent



- Given $U = p \in \mathcal{P}_{4} (\mathbb{F}) : p(6)= 0$, what does this mean in **laymans** terms and how would we express it **symbolically**?
	- For input value 6, the polynomial evaluates to 0
		- $p(6) = 0 = a_0 + a_1 z + a_2 z^2 + a_3 z^3 + a_4 z^4$ when $z = 6$
- Algebraically, how can we factorize $p(6) = 0$
	- $p(z) = (z - 6) q(z)$ for any polynomial $q(z)$, or $p(z) = (z - 6)(b_0 + b_1 z + b_2 z^2 + b_3 z^3$
- Given a polynomial root / divisibility condition like $p(6) = 0$, what form does the basis take?
	- Basis = $(z - 6) \cdot \{1, z, z^2, z^3 \}$, aka
	- $\{(z-6), z(z-6), z^2 (z-6), z^3 (z-6) \}$
- For subspace $U$ w/ linear constraint $p(6) = 0$, how to define complementary subspace W where $P_4(F) = U \oplus W$
	- $W = \operatorname{span}\{1\} = \{c : c \in \mathbb{F}\}$
	- So $p(6) = c$

- Given a polynomial derivative-evaluation condition like $p^{(k)}(6) = 0$, what form does the basis take?
	- Remove the k-th index term - say $p''$ or $p^{(2)}$, then  $\{1, (z-6), \operatorname{REMOVE}, (z-6)^3, ... \}$
- Given a polynomial derivative-evaluation condition like $p^{(k)}(6) = 0$, why do we remove the k-th term of the standard basis?
	- Consider: $p''(z) = 2 b_2 + 6 b_3 (z - 6) + 12 b_4 (z-6) + ...$
	- The only way for $p''(6) = 0$ is when $b_2 = 0$
	- So only polynomials without this term form a basis

- Given a polynomial of with the constraint $p(a)  = p(b)$, what form does the basis take?
	- $p(z) = \operatorname{constant} + (z-a)(z-b)r(z)$  where $r(z) \in \mathbb{P}_{2}(F)$ so its basis is $\{1, z, z^2\}$

- What are the two patterns of polynomial conditions?  e.g. $\mathcal{P}_n(\mathbb{F}) = \{... : p(6) = 0\}$
	- Root / divisibility conditions: $p(6) = 0$
	- Derivative-evaluation conditions $p^{(k)}(6) = 0$


- Sequence for defining basis from polynomials
	- 1. **Impose** constraints on coefficients symbolically (usually around 0)  
	- 2. **Solve** for dependent variables like $a_0 = - 2 a_2 - 5 a_4$
	- 3. **Substitute** back into original polynomial equation  
	- 4. **Apply** to standard basis

- How to polynomial-ify constraint $\int_{-h}^{h} p(z) dz$
	- Separate out each term:  
	- $\int_{-h}^{h} p(z) dz =$
		- $a_0 \int_{-h}^{h} 1 dz +$ (even $= 2(1) a_0$)  
		- $a_1 \int_{-h}^{h} z dz +$ (odd >> 0)  
		- $a_2 \int_{-h}^{h} z^2 dz + ...$ (even >> 2(...) a_2)

