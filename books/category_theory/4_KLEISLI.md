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
