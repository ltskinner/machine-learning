# Normal Equation - Original Notes

- `normal equation`
  - in Ax = b
    - x represents the `scaling factors` for each independent direction contributed by the columns of A
    - x only has meaning in the context of A
    - A contains structural and geometric significance
  - $\bar{b} - \bar{p}$ or $A\bar{x} - \bar{v}$ are orthogonal
  - $V^{T}(\bar{b} - \bar{p}) = \bar{0}  $
  - where $b = Vc$
    - b = Vc projects coefficents c onto the space spanned by the columns of V
    - b = V_{c -> b}c
    - V projects coordinates onto d-dimensional space
      - V projects coordinate vector c onto b
      - V is the projection matrix
    - c describes the coordinates of b in the n-dimensional basis defined by V
  - or
  - $A^{T}(A\bar{x} - \bar{v}) = \bar{0}  $
    - ->$A^{T}A\bar{x} - $A^{T}\bar{v} = $A^{T}\bar{0}  $
    - ->$A^{T}A\bar{x} = $A^{T}\bar{v}  $
    - ->$\bar{x} = (A^{T}A)^{-1}$A^{T}\bar{v}  $
  - where:
    - A is some basis (linearly independent matrix)
    - Ax is how you compute the point on the plane the basis represents that is closest to p/v
      - (from above - where b is the point, V is the basis, and c is the coefficents)
    - remember to distribute the A^T and then invert A^{T}A as (A^{T}A)^{-1} to solve for x
  - or
  - in optimization:
    - the vector $\bar{b} - A\bar{x}  $
    - joins $\bar{b}$ to its closest approximation:
      - $\bar{b}' = A\bar{x}'  $ on the hyperplane defined by the columns space of A
      - (which is orthogonal to the hyperplane and therefore every col of A)
    - bringing us to the `normal equation`
      - $A^T (\bar{b} - A\bar{x}) = \bar{0}  $, which yields
      - $\bar{x} = (A^{T}A)^{-1} A^{T}\bar{b} $
      - (assuming A^T A is invertible - easy to do when A is tall)