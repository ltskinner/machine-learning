Definition:

A list $v_1, ..., v_m$ if vectors in $V$ is called `linearly independent` if the only choice of $a_1, ..., a_m \in \mathbb{F}$ that makes
$a_1 v_1 + ... + a_m v_m = 0$
is $a_1 = ... = a_m = 0$

The empty list $()$ is also declared to be `linearly independent`

## Linearly DEPendent

A list of vectors in V is called `linearly dependent` if it is **NOT** `linearly independent`

In other words, a list $v_1, ..., v_m$ of vectors in V is linearly DEPendent if there exists $a_1, ..., a_m in \mathbb{F}$, not all 0, such that $a_1 v_1 + ... + a_m v_m = 0$

### Linear Dependence Lemma

Suppose $v_1, ..., v_m$ is a linearly dependent list in V
There exists $k \in \{1, 2, ..., m\}$ such that
$v_k \in \operatorname{span}(v_1, ..., v_{k-1})$

Furthermore, if $k$ satisfies the condition above and the kth term is removed from $v_1, .., v_m$, then the span of the remaining list equals $\operatorname{span}(v_1, ..., v_m)$

## Length of Linearly Independent List

In a finite-dimensional vector space, the length of *every* linearly independent list of vector is less than or equal to the length of every spanning list of vectors.


## Proof Template

To prove independence:  
- start with $a_1 v_1 + ... + a_n v_n = 0$
- show all coeffs $a_1 = ... = a_n = 0$



## Flashcards


- Formal **condition** for linearly **in**dependent
	- A list of vectors [$]v_1, ..., v_m[/$] in V is linearly independent if: 
	- Say we want to make [$]a_1 v_1 + ... a_m v _ m = 0[/$]. 
	- If the ONLY way to make this true is for all coefficients [$]a_1 = ... = a_m = 0[/$], then linear independence
- What does "two vectors are scalar multiples of each other" mean?
	- Scalar multiple = linear **dep**endence = reduce dimension
- When defining linear independence in R vs C, what does this do to coefficients $a_i$?
	- In R, the coeffs are all real numbers  
	- In C, the coeffs can be $i$ which makes for peculiar interactions
- When is a list of length 2 linearly independent?
	- IFF neither of the two vectors are scalar multiples of each other
- **Condition** for a linear **dep**endent list of vectors (formal)
	- If it is NOT linearly **in**_dependent_  
	- Aka if when [$]a_1 v_1 + ... + a_m v_m = 0[/$], some of the coefficients [$]a_i[/$] are not all 0
- Linear **dep**endence lemma
	- Given a linearly dependent list of vectors, there exists at least one vector that resides in the span of the "previous" vectors 
	- As such, this vector can be removed
- What is preserved under invertible [$]A^{-1}[/$] linear transforms?
	- Linear independence
- If [$]v_1, ..., v_m[/$] and [$]w_1, ..., w_m[/$] are linearly independent, is [$]v_1 + w_1, ..., v_m + w_m[/$] also linearly independent?
	- No it is not - "pairwise summing" of two independent lists may destroy indpendence completely because they could **cancel** each other 
	- Consider: [$]w_i = - v_i[/$] canceling each other, making the second list (0, ..., 0), which we already know is not linearly independent
- What makes v_1, …, v_m **linearly independent** (free text about span)
	- [$]v_1, ..., v_m[/$] is linearly indpendent IFF each vector in [$]\text{span}(v_1, ..., v_m)[/$] has exactly one representation as a **linear combination** of [$]v_1, ..., v_m[/$]
	- This is working toward a basis
- What is special about a list of vectors with the 0 vector
	- Every list with the 0 vector is linearly **dep**endent
	- B/c the 0 vector can be reached as a linear combination of other vectors
- Is a list of length 1 linearly independent?
	- IFF the vector in the list is **NOT** [$]0[/$]
- Length of linearly independent list [$]\leq[/$] length of spanning list
	- In a finite-dimensional vector space:  
	- The length of every linearly independent list of vectors is [$]leq[/$]  
	- The length of every spanning list of vectors
- If we are in R^3, what is the longest length of list that is linearly independent?
	- A list of length 3. Any greater will NOT be independent
- Is the empty list () linearly independent?
	- yes
