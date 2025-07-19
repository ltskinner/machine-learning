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
- Membership Check
  - Tests if a specific x belongs in the set
  - $x \in S $ where S is a set - note **NO** brackets
  - $\land, \lor, \lnot $ are all fair game at this level, just **NOT** $\exists, \forall $
- Quantified Logic
  - States a fact about **whether anything** satisifes a condition
  - Makes a global claim about:
    - some of x $\exists x \in D (P(x)) $
    - all of x $\forall x \in D (P(x)) $

## Skeletons

- quantificational logic
  - what conditions need \exists vs \forall
- when do \implies get involved

## Part 1: Flat Set Builder <-> Quantified Logic

- 1. Understand Set-Builder Syntax Semantics
- 2. Translate Set Membership into Quantified Logic
- 3. Move Between: Set-builder notation; Membership check expressions; Quantificational logic
- 4. Know when to use \exists vs \forall

## Part 2: Families of Sets

Goal: Develop fluency in nested set structures — sets whose elements are themselves sets (i.e. "families")

- 1. Understand What a Family Is
- 2. Interpret Membership in a Family
- 3. Learn Set Operations on Families
- 4. Recognize When Nested Logic Is Needed

## Part 3: Indexed Families of Sets

Goal: Learn how to handle collections of sets labeled by an index (i.e. think of them like arrays of sets)

- 1. Understand Indexed Notation
- 2. Decode Index-Based Set Ops
- 3. Use Quantifiers over the Index Set
- 4. Relate to Code-Like Thinking

## Part 4: Layer Switching and Translation

- 1. Recognize the Three Layers
- 2. Know What Each Layer Is For
- 3. Translate Across Layers
- 4. Prioritize Clarity Over Density
