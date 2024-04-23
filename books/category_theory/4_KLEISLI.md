# Chapter 4. Kleisli Categories

- pure function: a function that always produces the same result, given the same input, and have no side effects

One can also model 'side effects', aka non-pure functions

Example:

- a function that updates a global state
- a function that mutates a global state
  - there is a side effect other than the type interface

The aggregation of the log is no longer the concern of the individual functions. They produce thier own messages, which are then, externally, concatenated into a larger log.

## 4.1 The Writer Category

The idea of embellishing the return types of a bunch of functions in order to piggyback some additional functionality is very fruitful

- unchanged: Types remain as objects
- redefine: morphisms are embellished functions

oh, this is a monoid

- a neutral element
- a binary operation

### Recipie for composition

(of two morphisms in the new category we are constructing)

- 1. Execute the embellished function corresponding to the first morphism
- 2. Extract the first component of the result pair and pass it to the embellished function corresponding to the second morphism
- 3. Concatenate the second component (the string) of the first result and the second component (the string) of the second result
- 4. Return a new pair combining the first component of the final result with the concatenated string

## 4.2 Writer in Haskell

```hs
-- defining the Writer type
type Writer a = (a, String)
-- the type Writer is parameterized by a type variable `a`
-- and is equivalent to a pair of `a` and `String`
-- the syntax for pairs is minimal:
-- just two items in parentheses, separated by a comma

-- our morphisms are functions form an arbitrary type to some Writer type:
a -> Writer b

-- well declare the composition as a funny infix operator
-- sometimes called the "fish"
(>=>) :: (a -> Writer b) -> (b -> Writer c) -> (a -> Writer c)

-- infix operator
-- two arguments m1 and m2
-- result is a lambda fn of one argument x
-- the lambda is the backslash

m1 >==> m2 = \x ->
    let (y, s1) = m1 x
      (z, s2) = m2 y
    in (z, s1 ++ s2)

-- identify morphism
-- in this case, it happens to be named 'return'
return :: a -> Writer a
return x = (x, "")

-- the embelleshed functions
upCase :: String -> Writer String
upCase s = (map toUpper s, "upCase ")
-- the function `map`
-- applies the character function toUpper to the string s

toWords :: String -> Writer [String]
toWords s = (words s, "toWords ")

-- the composition of the two functions is accomplished with
-- the help of the fish operator
process :: String -> Writer [String]
process = upCase >=> toWords
```

## 4.3 Kleisli Categories

Kleisli Category: a category mased on a monad

- a Kleisli category has:
  - as objects: the types of the underlying programming language
  - morphisms from type A to type B
    - why not denote as Hom(a, b)
  - are functions that go from A to type derived from B
  - using the particular embellishment
- each Kleisli category defines its own way of composing such morphisms, as well as the identity morphisms with respect to that composition
- the imprecise term "embellishment" corresponds to the notion of an `endofunctor` in a cateogry
- the `Writer` is a particular monad
  - formally, the *writer monad*

## Challenge

- 1. Construct the Kleisli category for partial functions
  - Define composition
  - Define identity
- 2. Implement the embellished function safe_reciprocal that returns a valid reciprocal of its argument, if its different from zero
- 3. Compose the functions safe_root and safe_reciprocal to implement safe_root_reciprocal that calculates sqrt(1/x) whenever possible
