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

## 2.4 Why Do We Need a Mathematical Model

When programming, there is syntax and grammar. These aspects of the language are described using formal notation at the begining of the language spec. The meaning, or semantics, is harder to describe

It doesnt matter that programmers never perform formal proofs of correctness. We always "think" that we write correct programs. We are usually quite surprised when it doesnt work as expected

*Denotational semantics* is based on math. Here, every programming construct is given its mathematical interpretation. If you want to prove a property of a program, you just prove a mathematical theorum.

Being able to prove correctness is kindof nice

## 2.5 Pure and Dirty Functions

- *pure functions*: functions that always produce the same result given the same input and have no side effects
  - In a pure functional language like Haskell, all functions are pure
  - Becuase of that, its easier to give these languages denotational semantics and model them using category theory
  - Monads let us model all kinds of effects using only pure functions

## 2.6 Examples of Types

### Void

Once you realize that types are sets, you can think of v exotic types. Whats the type corresponding to an empty set? its not a Void, its a type thats not inhabited by any values. You can define a function that takes Void, but you can never call it. To call it, you would have to procide a value of the type Void, and there just arent any. As for what the function can return, there are no restructions whatsoever. It can return any type (althought it never will , because it cant be called). In other words its a function thats polymorphic in the return type

`absurd :: Void -> a` (remember, `a` is a type variable that can stand for any type)

The name is not coincidental - Curry Howard isomorphisms. Void represents falsity, and the type of the function `absurd` corresponds to the statement that "from falsity follows anything" aka "ex falso sequitur quodlibet"

### Singleton set

- type with only one value

```c
int f44() { return 44; }
```

- it looks like this function takes "nothing"
- a function that takes "nothing" can never be called, because there is no value representing nothing
- so, conceptually, it takes a dummy value of which there is only one instance ever, so we dont have to mention it explicitly
- in hs, there is a symbol for this: `()`
  - this also is the type

```hs
-- declares f44 takes type () (pronounced unit) into the type Integer
f44 :: () -> Integer
-- defines f44 by pattern matching the only constructor for unit
f44 () = 44

-- to call
f44 ()
```

You can use `_` like in python for variables you arent going to use

```hs
unit :: a -> ()
unit _ = ()
```

### Bool

```hs
data Bool = True | False
-- Bool is either True or False
```
