Pattern:
- Assume two items exist
- Show they are equal

## Template

To prove uniqueness of X:
- 1. Assume X and X' both satisfy the defining property
- 2. Use the axioms
- 3. Show X = X'

Done

## Examples

### 1.26 Unique Additive Identity

Definition: A vector space has a unique additive identity

Proof:
Suppose 0 and 0' are both `additive identities` for some vector space V
$0' = 0' + 0$ (holds because 0 is an `additive identity`)
$= 0 + 0'$ (holds because of `commutativity`)
$= 0$ (holds because $0'$ is an additive identity)
Thus $0' = 0$, proving that V has only one additive identity

### 1.27 Unique Additive Inverse

Definition: Every element in a vector space has a unique additive inverse

Proof:
Suppose V is a vector space
Let $v \in V$
Suppose $w$ and $w'$ are additive inverses of $v$, then
$w = w + 0$ (introduce additive identity)
$= w + (v + w')$ ($v + w' = 0$, because $w'$ is an additive inverse $w' = -v$)
$= (w + v) + w'$  (associativity)
$= 0 + w'$ (additive inverse, $w + v = 0$, $w = -v$)
$= w'$ (additive identity)

Thus $w = w'$, as desired