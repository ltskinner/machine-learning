# Category Theory for Programmers

- [Original Blog Post](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/)
- [Non-Kindle Hardcover on Amazon](https://www.amazon.com/Category-Theory-Programmers-Bartosz-Milewski/dp/0464243874/ref=sr_1_1)
- [pdf github repo](https://github.com/hmemcpy/milewski-ctfp-pdf)
- [pdf releases](https://github.com/hmemcpy/milewski-ctfp-pdf/releases/tag/v1.3.0)

## Preface

".. to convince you that this book is written for you, whatever objections you might have to learning one of the `most abstract branches of mathematics` in your 'copious space time' are totally unfounded"

- Category theory is a treasure trove of extremely useful programming ideas
  - Haskell programmers have been tapping this resource for a long time, other languages are picking it up, but it could be going faster
- There are many different kinds of math, that appeal to different audiences
  - Argue that category theory is the kind of math that is well suited for programmers
  - It deals with the kind of structure that makes programs composable
- `Composition` is at the root of category theory - and is part of the definition of the `category` itself

"If you're an experienced programmer, you might be asking yourself: Ive been coding for so long without worrying about category theory or funcitonal methods, so whats changed? Surely you cant hlep but ontice that theres been a steady stream of new functional features invading imperative languages. Even Java, the bastion of object-oriented programming, let the lambdas in. C__ has recently been evolving at a frantic pace - a new standard every few years - trying to catch up with the changing world. All this activity is in preparation for a disrubptive change change or, as we physicists call it, a phase transition. If you keep heading water, it will eventually start boiling. We are now in the position of a frog that must decide if it should continue swimming in increasingly hot water, or start looking for some alternatives."

"Just like the builders of Europe great gothic cathedrals we've been honing our craft to the limits of material and structure... we've done all this based on very flimsy theoretical foundations. We have to fix those foundations if we want to move forward"

- Multicore revolution is big
  - oop doesnt take advantage of this
  - data hiding, the basic premise of oop, when combined with sharing an dmutation, becomes a recipie for data races

## Part 1

### [1. Category: The Essence of Composition](./1_CATEGORY.md)

"Neutral values like zero or id are extremely useful when working with symbolic variables"

- A category consists of objects and arrows (morphisms)
- Arrows can be composed
- The composition is associative (f.id=f, id.f=f)
- every object has an identity arrow that serves as a unit under composition

### [2. Types and Functions](./2_TYPES_FNS.md)
