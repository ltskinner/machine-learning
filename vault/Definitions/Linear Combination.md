
Definition:
A `linear combination` of a list $v_1, ..., v_m$ of vectors in V is a vector of the form:
$a_1 v_1 + ... + a_m v_m$
Where $a_1, ..., a_m \in \mathbb{F}$

## Example

(17, -4, 2) is a `linear combination` of (2, 1, -3) and (1, -2, 4) because

(17, -4, 2) = 6(2, 1, -3) + 5(1, -2, 4)

or:
- 17 = 6(2) + 5(1)
- -4 = 6(1) + 5(-2)
- 2 = 6(-3) + 5(4)


## Flashcards

- What is a **linear combination** of scalar multiples against a list of vectors? (Main definition)
	- The **linear combination** of a **list** $v_1, ..., v_m$ **of vectors** in V is a vector of the **form**:
	- $a_1 v_1 + ... + a_m v_m$
	- where $a_1, ..., a_m \in F$
- When given a resulting linear combination and the list of "input" vectors - what do we need to find?
	- The coefficients $a_1, ..., a_m \in F$ such that a linear combination is possible



- Give an example of a non-trivial coefficient of **linear combination** for a vector space over C (as opposed to R)
	- $(i)(1 + i)$ or $(-i)(1 - i)$


- - What does $F^S$ denote?
	- If S is a set, then F^S denotes the set of functions from S to $\mathbb{F}$
	- Aka the set of functions from inputs in the **domain of S**, to the **codomain** of $\mathbb{F}$
- In laymans terms, what does $\mathbb{F}^S$ express?
	- $\mathbb{F}^S$ is "all the ways to assign a scalar value to each element of S"
- What is the **sum** of two functions $f, g \in \mathbb{F}^S$
	- $(f + g)(s) = f(s) + g(s)$
- What is the product of scalar multiplication ($\lambda$) on a function $f \in \mathbb{F}^S$
	- For $\lambda \in \mathbb{F}, f \in \mathbb{F}^S$  For all $s \in S$:  $(\lambda f)(s) = \lambda f (s)$

- How do we express **addition on differentiable functions** for linear combinations? (formal)
	- $(f + g)'(s) = f'(s) + g'(s)$
- How do we express **scalar multiplication on** derivatives for linear combinations? (formal)
	- $(\lambda f)'(s) = \lambda f'(s)$
-
- Dude, literally, what is linearity?
	- It is NOT "is a line". It literally is (for linear operator [$]L[/$]), the operator has:  
		- Addition: [$]L(u + v) = T(u) + T(v)[/$]  
		- Scalar multiplication [$]L(\lambda u) = \lambda L(u)[/$]
- What is [$]2 a_0 + \frac{2}{3} a_2 + \frac{2}{5} a_4[/$] called? And why can we solve for [$]a_0[/$] "arbitrarily"?
	- This is a **linear constraint** 
	- Linear constraint means one variable *must* depend on the others - we solve for [$]a_0[/$] because its the cleanest

- Why are straight lines linear?
	- Because they obviously respect operational constraints of:  
		- addition  
		- scalar multiplication
