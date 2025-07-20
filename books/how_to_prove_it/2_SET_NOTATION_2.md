# Set Notation - From the Ground Up

Three Layers of Notation:

- Set-Builder / SetDefinition
  - Defines a set
  - comes in two (equivalent) styles
    - `bounded` (domain-first)
      - $\{x \in Z | P(x) \} $
      - "Take $x \in Z $ such that x satisfies Property"
      - "Filter down from a known universe Z"
      - When to use:
        - to highlight the *type* of object working with
        - for clarity and emphasis
      - **dont over-read into this**
    - `unbounded` (condition-only)
      - $\{x | x \in Z \land P(x) \} $
      - "Let x be anything, and only keep if $x \in Z $ and P(x)"
  - $\{x \in D | P(x) \} $
  - for families, will see quantifiers $\forall, \exists $ in set
- Membership Check
  - Tests if a specific x belongs in the set
  - $x \in S $ where S is a set - note **NO** brackets
  - $\land, \lor, \lnot $ are all fair game at this level, just **NOT** $\exists, \forall $
- Quantified Logic
  - States a fact about **whether anything** satisifes a condition
  - Makes a global claim about:
    - some of x $\exists x \in D (P(x)) $
    - all of x $\forall x \in D (P(x)) $
  - **All** variables must be defined in terms of a $\forall $ or $\exists in \mathbb{X} $
    - Unless we are coming from membership check form and the variable being checked has already been defined

## On Families

- the basic structure in set form is:
  - $F = \{ A \subseteq \mathbb{Z} | $
    - (rules about set A meta properties)
    - $\land $
    - (rules governing each element x of A)
    - $\} $
  - all equivalent:
    - $F = \{ A \subseteq Z | \forall x \in A (Even(x)) \} $
    - $F = \{ A \subseteq Z | \forall x \in A, \exists n \in Z (x = 2n)) \} $

## Skeletons

- quantificational logic
  - what conditions need \exists vs \forall
- when do \implies get involved
- going from plain text desc to:
  - set builder
  - to quantified

## Part 1: Flat Set Builder <-> Quantified Logic

- [x] 1. Understand Set-Builder Syntax Semantics
- [x] 2. Translate Set Membership into Quantified Logic
- [x] 3. Move Between: Set-builder notation; Membership check expressions; Quantificational logic
- [x] 4. Know when to use \exists vs \forall

## Part 2: Families of Sets

Goal: Develop fluency in nested set structures — sets whose elements are themselves sets (i.e. "families")

- [x] 1. Understand What a Family Is
- [x] 2. Interpret Membership in a Family
- [x] 3. Learn Set Operations on Families
- [x] 4. Recognize When Nested Logic Is Needed

## Part 3: Indexed Families of Sets

Goal: Learn how to handle collections of sets labeled by an index (i.e. think of them like arrays of sets)

- [x] 1. Understand Indexed Notation
- [x] 2. Decode Index-Based Set Ops
- [x] 3. Use Quantifiers over the Index Set
- [x] 4. Relate to Code-Like Thinking

## Part 4: Layer Switching and Translation

- 1. Recognize the Three Layers
- 2. Know What Each Layer Is For
- 3. Translate Across Layers
- 4. Prioritize Clarity Over Density
