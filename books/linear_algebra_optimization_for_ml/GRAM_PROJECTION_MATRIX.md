# Gram Matrix

Computes the **pairwise dot products** of vectors

Two flavors:

- `left Gram matrix`
  - $AA^T$ of the `row space` of A
  - encodes relationships between rows of A
- `(right) Gram matrix`
  - $A^T A$ of the `column space` of A
  - encodes relationships between cols of A

Properties:

- `symmetric`
  - dot products are `commutative`
  - v_{a} \cdot v_{b} = v_{b} \cdot v_{a}
- `positive semi-definite`
  - all `eigenvalues` are non-negative
- `rank`
  - rank(AA^T) = rank(A^T A) = rank(A)

Geometry:

Entries measure angles and lengths between vectors

- Diagonal entries
  - $(AA^T)[i, i] = \|a_{i}a_{i}\|^{2} $
  - Squared norm (length) of the i-th row
  - the *magnitude* of each row vector
- Off-diagonal entries
  - $(AA^T)[i, j]$
  - The `cosine similarity` (scaled by the norms)
  - how much `constraints` "overlap"
    - measures how aligned the rows are
    - the *pairwise alignment (orthogonality)* between each row vector
  - represents the covariance matrix, capturing:
    - variability
    - relationships between constraints

In relation to Projection Matrices `P`

$P = A(A^T A)^{-1} A^T$

- $(A^T A)$
  - `(right) Gram matrix`
  - ensures projection resepcts geometry of the `columns`
  - pairwise relationship between columns
- $(A^T A)^{-1} $
  - the inverse normalizes the contributions of each column to avoid distortion
- $A...A^T  $
  - `left Gram matrix`
  - the constraints of the rows map to the column space
  - A^T maps the trailing `b` from R^n to R^d
  - A guarantees the result lies in the column space of A

Looking at P = QQ^T:

- we know that:
  - $\|P\bar{b}\| \leq \|\bar{b}\| $
  - this is based on
  - $\|QQ^T \bar{b}\| = \|Q^T \bar{b}\|  $
  - here:
    - $\|Q^T \bar{b}\| $ already applies all the constraints to b
      - but within the subspace R^d
    - the leading Q just reconstructs the vector in the output dimenion subspace R^n
