# Set Notation

Three layers of notation:

- `set-builder` / `elementhood test notation` / `set theory notation`
  - $\{n^2 | n \in \mathbb{N} \} $
    - sameas: $n^2 \land n \in \mathbb{N} $
    - domain of values | condition (such that passed this filter) (elementhood test)
  - **python:** `[n**2 for n in natural_numbers]`
  - describes a set
  - human-friendly *structure*
- `membership logic`
  - $x \in \{n^2 | n \in \mathbb{N} \} $
  - **python:** `x in [n**2 for n in natural_numbers]`
  - **python:** `any([x == n**2 for n in natural_numbers])`
  - asserts an element is in a set
  - "x is an element in the above set"
- `expanded logic`
  - $\exists n \in \mathbb{N} (x = n^2) $
    - literally use the $n \in \mathbb{N}$
    - and then introduce $x$ within the $( x = ...) $
  - this is **pure logic**
    - sometimes youll use $\exists $
    - other times just $ \land \lor$
  - used in formal proofs or predicate logic derivations
  - for *symbolic manipulation* in proofs

## Heuristics

- $x \in \{ ... \} $
  - >> a claim is being made here
  - >> truth valued statement
  - Ask: "Am I saying something about x or am i defining something new"
- No $\in $ before $\{ ... \} $
  - >> just defining a set here
- Three:
  - Am I describing something?
    - >> set builder notation
  - Am I testing membership
    - >> membership logic notation
  - Am I explaining something?
    - >> expanded logic

## Interchange Patterns

- $x \in A \cup B $
  - $x \in A \lor x \in B $
- $x \in A \cap B $
  - $x \in A \land x in B $
- $x \in \{ f(n) | n \in N \} $
  - $\exists x \in N (x = f(n)) $
  - n is a *generator*
- $x \subseteq P(A) $
  - $\forall y (y \in x \implies y \in A) $
  - x is a set
  - y is any element in x
- $x = P(A) $
  - $\forall y (y \in x \iff y \subseteq A) $
  - $\forall y (y \in x \iff \forall y (y \in x \implies y \in A)) $
- $x \in \mathbb{P}(A) $
  - $x \subseteq A $
