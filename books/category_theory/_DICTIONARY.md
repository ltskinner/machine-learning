# Dictionary

- `category`
  - consists of `objects` and `arrows` that go between them
  - the essence of a category is `composition`
  - or, the essence of `composition` is a category
  - formal def: consists of 3 mathematical entities
    - objects
    - morphisms (maps, arrows)
    - composition (associativity, identity)
- object
- arrow
  - also called `morphisms`
  - are like `functions`
- `identity`
  - an `equality` relating one mathematical `expression` A to another mathematical expression B, such that A and B (which may contain some variables) produce the same value for all values of the variables within a certain range of validity
  - A = B is an identity if A and B define the same functions, and an identity is an equality between functions that are differently defined
  - identities are the tripple bar, like an equal sign but with 3
- equality
- expression
- variables
- universal quantification
- `identity morphisms`
- monoid
  - a set with a binary operation
    - the operation must be associative
  - there must be a special element that behaves like a `unit` wrt the operation
  - binary operation -> function -> morphism
  - *single object* category, with a set of morphisms that follow appropriate rules of composition
  - categorical monoid (one object category)
- binary operations
- unit
  - the id composition is a unit arrow
  - unit is like itself
  - re: f44 :: () -> unit
    - there are no args, there is just a unit
- associativity
  - (a+b)+c = a+(b+c)
  - aka - we can skip the parentheses when adding numbers
- cartesian product
  - AxB = {(a, b) | a E A and b E B}
  - set of all ordered pairs (a, b) where a is in A and b is in B
    - A = {x, y, z}
    - B = {1, 2, 3}
    - AxB:
      - (x, 1), (x, 2), (x, 3)
      - (y, 1), (y, 2), (y, 3)
      - (z, 1), (z, 2), (z, 3)
    - these are ordered pairs: the order matters
      - some operations (0, 3) v diff than (3, 0)
        - like: a / b
- monad
  - monads are monoids in the category of endofunctors
  - a monad is one example of a monoid
- endofunctor
  - a functor from a category to itself
- functor
  - a homomorphism of categories
- homomorphism
  - a funciton between `structured sets` that preserves whatever structure there is around
- magma
  - a binary operation on a set S is a function (-).(-):SxS->S
    - from the Cartesian product SxS to S
  - magma:
    - aka binary algebraic structur
    - aka mono-binary algebra
  - is a set equipped with a binary operation on it
  - uhh:
    - a monoid may be thought of as:
      - a magma with associativity and identity

### Orders

- homset
  - A set of morphism from object a to object b in category C is called a *hom-set* written as C(a, b) or, Hom_C(a, b)
  - every homset in a preorder is either empty or a singleton
  - M(a,b) is like a representation of every possible function(morphism) that we could take on the path of a to b
- preorder
  - a <= b and b <= c, so a <= c
  - "thin" category
  - you may have cycles in a preorder
- partial order
  - a <= b, and b <= a, so a == b
  - cycles are forbidden in a partial order
- linear order aka total order
  - condition that any two objects are in a relation with each other, one way or another

## Graphs

- directed graph

## CS

- polymorphic function
  - a function that can evaluate to or be applied to values of different types
- pure function
  - a functions that always produce the same result given the same input and have no side effects
- non-pure function
  - a function that has side effects
  - there is a "side effect" or effective "output" other than the type interface
- partial function:
  - a function not defined for all possible values of its argument
  - note: its not really a function in a mathematical sense
