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

p :: (Int, Int, Bool) -> Int
p (x, _, _) = x

q :: (Int, Int, Bool) -> Bool
q (_, _, b) = b
```

What if we were to explore the ranking (another part of the universal construction). We want to be able to compare two instances of our pattern

We want to compare one candidate object:

- `c` and its two projections `p` and `q`
  - with another object
- `c'` and its two projections `p'` and `q'`

We would like to say that `c` is "better" than `c'` if there is a morphism `m` from c' to c - but thats too weak. We also want its projections to be "better" or "more universal" than the projections of c'. What it means that the projections p' and q' can be reconstructed from p and q using m

```hs
p' = p . m
q' = q . m
```

Another way of looking at these equatinos is that m `factorizes` p' and q'. Just pretend that these equatinos are in natural numbers and the dot is multiplicaiton: m is a common factor shared by p' and q'

Example with diagrams:

```hs
-- mapping m for the first candidate
m :: Int -> (Int, Bool)
m x = (x, True)

-- the two projections, p and q, can be reconstructed as
p x = fst (m x) = x
q x = snd (m x) = True

-- the m for the second example is similarly uniquely determined:

m (x, _, b) = (x, b)
```

We were able to show that (Int, Bool) is better than either of the two candidates

Lets see why the opposite is **not** true. Could we find some m' that would help us reconstruct fst and snd from p and q

```hs
fst = p . m'
snd = q . m'
```

In the first example, q always returned True and (yet) we know that there are pairs whose second component is False. We cannot reconstruct snd from q because there are pairs that return False

Second example:

This is different because we retain information after running either p or q, but there is more than one way to factorize fst and snd. Because both p and q ignore the second component of the tripke, our m' can put anything in it. We can have:

```hs
m' (x, b) = (x, x, b)
-- or
m' (x, b) = (x, 42, b)
```

Putting it all together: given any type c with two projections p and q, there is a unique m from c to the Cartesian product (a, b) that factorizes them. In fact, it just combines p and q into a pair

```hs
m :: c -> (a, b)
m x = (p x, q x)
```

That makes the Cartesian product (a, b) our best match, which means that this universal construction works in the category of sets. It picks the product of any two sets

Now, forget about sets, and define a product of two objects in any category using the same universal construction. Such a product doesnt always exist, but when it does, it is unique up to a uniqu isomorphism

"A product of two objects a and b is the object c equipped with two projections such that for any other object c' equipped with two projections there is a unique morphism m from c' to c that factorizes those projections"

A (higher order) function that produces the factorizing function m from two candidates is sometimes called the factorizer. In our case, it would be the function:

```hs
factorizer :: (c -> a) -> (c -> b) -> (c -> (a, b))
factorizer p q = \x -> (p x, q x)
```

## 5.6 Coproduct
