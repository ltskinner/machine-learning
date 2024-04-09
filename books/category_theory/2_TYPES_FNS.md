# Chapter 2. Types and functions

## 2.1 Who Needs Types?

- static vs dynamic
- strong vs weak typing

type checking provides another barrier against nonsensical programs

- dynamically typed languages: type mismatches would be discovered at runtime
- strongly typed statically checked languages type mismatches are discovered at compile time, eliminating lots of incorrect programs before they have a chance to run

The usual goal in the typing monkeys thought experiement is the production of the complete works of shakespeare. Having a spell checker and a grammar checker in the loop would dramatically increase the odds. The analog of a type checker would go even further by making sure that, once Romeo is declared a human being, he doesnt sprout leaves or trap photons

## 2.2 Types Are About Composability

Category theory is about composing arrows. But not any two arrows can be composed. The target object of one arrow must be the same as the source object of the next arrow. The stronger the type system of the language, the better this match can be described and mechanically verified

"Unit testing may catch some of the mismatches, but testing is almost always a probabilistic rather than a deterministic process. Testing is a poor substitute for proof"

## 2.3 What Are Types?

Types are a set of values

- Bool: {True, False}
- Char: {a, b, c, ....}
- String: infinite set

There is a catrgory of sets, called **Set**. In **Set**, objects are sets and morphisms (arrows) are functions.

- Set is a very special category because we can peek inside its objects and get lots of intuitions
- we know functions map elements of one set to elements of another set
  - they can map two elements to one
  - but not one element to two
- we know that an identity function maps each elmenet of a set to itself
- The plan is to graudally forget all this information and instead express all those notions in purely categorical terms
  - aka interms of objects and arrows

There are some calculations that involve recursion, and some that never terminate.

We cant ban non-terminating functions from Haskell because distinguishing between terminating and non-terminating functions is un-decidable - this is known as the "halting problem". To address this, every type is extended by a special value called the *bottom* which basically is the result of anon-terminating computation

So `f :: Bool -> Bool` = `{True, False, _|_}`

When you accept this bottom as part of the type system, it is convenient to treat every runtime error as a bottom, and even allow functions to return the bottom explicitly (this is usually done using the expression *undefined*)

```hs
f :: Bool -> Bool
f x = undefined
```

This definition type checks because undefined evaluates to bottom, which is a momber of any type, including Bool

Functions that may return bottom are called *partial* as opposed to *total* functions, which return valid results for every possible argument.

Because of the bototm, youll see the category of Haskell types and functions referred to as *Hask* rather than *Set*. From the pragmatic point of view, it is ok to ignore non-terminating functions and bottoms, and treat Hask as a bona fide Set
