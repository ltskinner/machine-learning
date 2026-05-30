Definition:
A `basis` is a list of vectors in V that is:
- [[Linearly Independent]], and
- [[Span]]s V

## Criterion for Basis

A list $v_1, ..., v_n$ of vectors in $V$ is a `basis` of $V$ IFF every $v \in V$ can be written uniquely in the form
$v = a_1 v_1 + ... + a_n v_n$
where $a_1, ..., a_n \in \mathbb{F}$

### Every Spanning list contains a Basis

Every spanning list in a [[Vector Space]] can be **reduced** to a `basis` of the vector space

### Basis of finite-dimensional vector space

Every [[Finite-Dimensional Vector Space]] has a `basis`

### Every linearly independent list extends to a basis

Every `linearly independent` list of vectors in a `finite-dimensional vector space` can be **extended** to a `basis` of the vector space

### Every subspace of V is part of a direct sum equal to V

Suppose $V$ is a finite-dimensional and $U$ is a subspace of $V$.
Then there is a subspace $W$ of $V$ such that $V = U \oplus W$


## Examples

### Defining a basis

Consider subspace U defined as:  
[$]U = \{(z_1, z_2, z_3, z_4, z_5 \in F^5 :[/$]  
[$]6 z_1 = z_2 and z_3 + 2 z_4 + 3 z_5 = 0 \}[/$]  

Declare the basis of U

Basically, burn these down to pattern of $x(...), y(...), z(...)$  
In this case the basis is: $\{$  
$(1, 6, 0, 0, 0),$  
$(0, 0, -2, 1, 0),$  cross term for b of (-2b - 3c)  
$(0, 0 -3, 0, 1)\}$ cross term for c of (-2b - 3c)

## Flashcards



- Formal **criterion** for basis (formal)
	- A list of vectors [$]v_1, ..., v_n[/$] in [$]V[/$] is a **basis** of [$]V[/$] IFF:  
	- Every [$]v \in V[/$] can be written uniquely in the form:  
	- [$]v = a_1 v_1 + ... + a_n v_n[/$]  
	- Where [$]a_1, ..., a_n \in \mathbb{F}[/$]
- What is the conceptual thru-line from sum of subspaces, to direct sum, thru linear combination, to linear independence, to basis?
	- Showing uniqueness by virtue of the only way to uniquely write some list of vectors subject to linear operations = 0 is when coefficients, or the vectors themselves are = 0  
	- Sum of Subspaces [$]v \in V \equiv v = v_1 + ... + v_m[/$] for some [$]v_1 \in V_1, ..., v_m \in V_m[/$]
	- Direct sum: [$]v_1 + ... + v_m = 0[/$] when [$]v_k = 0[/$]  
	- Linear Independence [$]\equiv[/$] Linear Combination = 0:  
	- [$]\rightarrow a_1 v_1 + ... + a_m v_m = 0[/$]  
	- Basis [$]\forall v \in V[/$] as [$]v = a_1 v_1 + ... + a_m v_m[/$]
- Length of basis wrt F^n ?
	- [$]F^n = n[/$] - any two **bases** of a finite-dimensional vector space have the same length
- What is the **standard basis** of [$]F^n[/$]
	- [$](1, 0, ..., 0), (0, 1, 0, ..., 0),[/$] 
	- [$]..., (0, ..., 0, 1)[/$]
- Definition of basis of V
	- A basis of V is a list of vectors in V that is:  
	- linearly independent  
	- spans V
- Every linearly independent list of vectors can be extended to what?
	- Basically: add new linearly independent vectors until spans the finite-dimensional vector space to produce a basis
	- If not already a basis, every linearly independent list of vectors in a finite-dimensional vector space can be extended to a basis of the vector space
- What is the relation between finite-dimensional vector spaces and a basis?
	- Every finite-dimensional vector space has a basis
- Given every list that spans a vector space, what can you do to convert them into a basis?
	- Reduce the list to just linearly independent components will make it a basis

	