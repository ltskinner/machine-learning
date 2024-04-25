# Products and Coproducts

"Every man is like the company he is wont to keep" - we are defined by our relationships

- `universal construction` - defines objects in terms of their relationships
  - one way:
    - pick a pattern, a particular shape constructed from objects and morphisms, and look for all its occurences in the category
    - if its a common enough pattern, and the category is large, chances are there will be many hits
    - Trick is to establish some kind of ranking among the hits, and pick what could be considered the best fit
- This is a lot like web searches. A query is like a pattern
  - a very general query will give you large `recall`: lots of hits
  - refine the query to increase its `precision`

## 5.1 Initial Object

- The simplest shape is a single object
  - there are many instances of these in a given a category
  - we need to establish some kind of rank
  - the only means at our disposal are morphisms
    - there may be an overall net flow of arrows from one end of the category to another (true for ordered categories)
    - generalize that notion of object precedence by saying that object a is "more initial" than object b, if there is an arrow going from a to b
    - the initial object is defined as one that has arrows going to all other objects
      - note there is no guarantee of this
      - bigger problem is too many objects
      - address using ordered categories: they allow at most one arrow between any two objects

The `initial object` is the object that has one and only one morphism going to any object in the category

- note, this doesnt guarantee the uniqueness of the initial object
- guarantees the next best thing:
  - uniqueness up to isomorphism

in a partially ordered set (*poset*), the initial object is its `least element` aka element <<<<<<<<=

- Some posets dont have an initial set:
  - set of all integers, positive and negative with <= relation for morphisms

In the category of sets and functions, the initial object is an empty set

- Remember, an empty set corresponds to the Haskell type `Void`
- The unique polymorphic function from `Void` to any other type is called `absurd`

```hs
absurd :: Void -> a
```

### 5.2 Terminal Object

Continuing with the single-object pattern, but change the way we rank objects

- object a is "more terminal" than object b if there is a morphism going from b to a (notice the reversal of direction)

A `terminal object` is the object with one and only one morphism coming to it from any object in the category

- also unique, up to isomorphism
- in a poset, the terminal object (if it exists) is the biggest object
- in the category of sets, the terminal object is a singleton
  - singletons = `unit` type `()` in Haskell

```hs
-- there is one and only one pure function
-- from any type to the unit type
unit :: a -> ()
unit _ = ()
```

- Note: the uniqueness condition is cruicial
- There are other sets (all others sets, actually, except for the empty set) that have incoming morphisms from every set

Example:

```hs
-- there is a Boolean-valued function (a predicate)
-- defined for every type
yes :: a -> Bool
yes _ = True
```

But `Bool` is not a terminal object. There is at least one more Bool-valued function from every type (except `Void`, for which both functions are equal to absurd)

```hs
no :: a -> Bool
no _ = False
```

Insisting on uniqueness gives us just the right precision to narrow down the definition of the terminal object to just one type

## 5.3 Duality

The only difference between initial and terminal object is the direction of the morphisms

For any category C we can define the `opposite category` C^op just by reversing the arrows

The opposite category automatically satisfies all the requirements of a category, as long as we simultaneously redefine composition

- Original:
  - f :: a -> b
  - g :: b -> c
  - h :: a -> c with h = g.f
- Reversed:
  - f^op :: b -> a
  - g^op :: c -> b
  - h^op :: c -> a with h^op = f^op.g^op

- Duality is a very important property of categories because it doubles the productivity of every mathematician working in category theory
- For every construction you come up with, there is its opposite
- For every theorum you prove, you get one for free

Constructions in the opposite category are often prefixed with "co"

- products, and coproducts
- monads and comonads
- cones and cocones
- limits and colimits
- reversing a co{} results in the original

The terminal object is the initial object in the opposite category

## 5.4 Isomorphisms

We understand the inverse in terms of composition and identity

```hs
f . g = id
g . f = id
```

Saying that the initial (terminal) object was "unique up to isomorphism", what is meant is that any two initial (terminal) objects are isomorphic

Universal constructions have an important property: "uniqueness up to unique isomorphisms"

## 5.5 Products

"The next universal construction"

- We know what a Cartesian product of two sets is: its a set of pairs
- But whats the pattern that connects the product set with its constituent sets?
- If we can figure that out, we'll be able to generalize it to other categories

All we can say:

- there are two functions (the projections) from the product to each of the constituents
- in Haskell, these two functions are called fst and snd and they pick, respectively, the first and the second component of a pair

```hs
fst :: (a, b) -> a
fst :: (x, y) = x

snd :: (a, b) -> b
snd (x, y) = y

-- simplified definitions using wildcards
fst (x, _) = x
snd (_, y) = y
```

Equipped with this seemingly limited knowledge, lets try to define a pattern of objects and morphisms in the category of sets that will lead us to the construction of a product of two sets, a and b

```hs
p :: c -> a
q :: c -> b

-- all cs that fit this pattern
-- will be considered candidates for the product
-- there may be lots of them
```

What if we were to explore the ranking (another part of the universal construction). We want to be able to compare two instances of our pattern
