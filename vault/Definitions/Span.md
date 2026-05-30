Definition
The set of all [[Linear Combination]]s of a list of vectors $v_1, ..., v_m$ in $V$ is called the `span` of $v_1, ..., v_m$, denoted by $\operatorname{span}(v_1, ..., v_m)$

In other words:
$\operatorname{span}(v_1, ..., v_m) = \{a_1 v_1 + ... + a_m v_m : a_1, ..., a_m \in \mathbb{F}\}$

The span of the empty list $()$ is defined to be $\{0\}$

Further:
The span of a list of vectors in [[Vector Space]] V is the smallest [[Subspace]] of V containing all vectors in the list

By spanning, the vector space V must be a [[Finite-Dimensional Vector Space]]

## Spans

If $\operatorname{span}(v_1, ..., v_m)$ equals V, we say that the list $v_1, ..., v_m$ `spans` V


## Proof Templates

### Proof template for showing two **spans** are equal?

We do this to show that two subspaces are equal

1. Show [$]\operatorname{span}(v_1, ..., v_m) \subseteq \operatorname{span}(w_1, ..., w_m)[/$]  
2. Show [$]\operatorname{span}(w_1, ..., w_m) \subseteq \operatorname{span}(v_1, ..., v_m)[/$]


### Verbiage for claiming w defined in terms of v is a span

"Because each [$]w_k[/$] is a **linear combination** of [$]v_1, .., v_m[/$], we have [$]\operatorname{span}(w_1, ..., w_k) \subseteq \operatorname{span}(v_1, ..., v_k)[/$]"


### Given a list with some variables we dont like, such as $v_1 + w, ..., v_m + w$ what do we do with this shit?

- Reexpress the list in terms of another list that resides in the same span  
- Components must be constructed from terms in the ugly list  
- Do this in a way that cancels the unknown terms  
- For every unknown variable, expect to lose a dimension  
- $v_1 + w, ..., v_m + w = ((v_1 + w) - (v_1 + w), ..., (v_m + w) - (v_1 + w))$
- $v_1 + w, ..., v_m + w = (0, ..., (v_m + w) - (v_1 + w)) = (0, ..., v_m - v_1)$






## Flashcards

- What is the definition of **the** **span** of a list of vectors?
	- The set of _all_ **linear combinations** of a list of vectors
	- This is called the **span** of [$]v_1, ..., v_m[/$] in V, which is denoted as [$]\operatorname{span}(v_1, ..., v_m)[/$].
	- Aka [$]\operatorname{span}(v_1, ..., v_m) = \{a_1 v_1 + ... + a_m v_m : a_1, ..., a_m \in F \}[/$]
- Give an alternate definition of span wrt subspaces?
	- The span (of a list of vectors in V) is:
	- the **smallest subspace of V** containing **all** vectors in the list (of vectors)
- What is the definition of span**s**? (span-s wrt V)
	- If [$]\operatorname{span}(v_1, ..., v_m) = V[/$], then the list [$](v_1, ..., v_m)[/$] **spans** V
- What is the span of the empty list? [$]()[/$] (no vectors in the list)
	- The span of [$]()[/$] is [$]\{0\}[/$]
- What is the minimum dimension of a linearly independent list to span [$]R^n[/$]?
	- the list must be of length n
- Say we have some vector [$]w = a_1 v_1 + ... + a_m v_m[/$], what does this tell us about where w resides?
	- Because vector [$]w[/$] is defined in terms of linear combinations of a list of vectors, [$]w[/$] must reside in the [$]\operatorname{span}(v_1, ..., v_m)[/$]  
	- (this is literally the definition of **span**)
- Is every spanning list in a vector space V a basis?
	- No, because the list may not be linearly independent
- Consider [$]U = \operatorname{Im}(T)[/$] where [$]T: F^2 \rightarrow F^5[/$]
	- Denote the requirement for a direct sum [$]\oplus[/$] of a subspace to span [$]F^5[/$]