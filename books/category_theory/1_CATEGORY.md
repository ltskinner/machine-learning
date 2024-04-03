# Chapter 1: The Essence of Composition

If you have an arrow from A to B, and an arrow from B to C, then there must be an arrow - their composition - that goes from A to C

The diagram is not a full category because its missing `identity morphisms`

## 1.1 Arrows as Functions

- `arrows` are also called `morphisms` which are like `functions`
- You have a function f that takes an argument of type A and returns a B
- You have another function g that takes B and return C
- you can compose them by passing the result of f to g
  - which defines a new function that takes A and returns C
- this is denoted by like g o f (lil hollow dot half way thru)
  - g o f
  - right to left order of composition
  - in mathematics and in Haskell functions compose right to left
  - "g after f"

C

```c
// function f
// argument a of type A
// returns type B
B f(A a);

// function g
// returns type C
// takes type B, as a
C g(B b);

// composition
C g_after_f(a) {
  g(f(a));
}
```

Haskell

```hs
-- function f :: input type A -> returns type B
-- note :: means "has the type of..."
-- the function type is created by inserting an arrow between two types
f :: A -> B

-- function g :: input type B -> returns type C
g :: B -> C

-- composition
g . f
```

## 1.2 Properties of Composition

There are two *extremely* important properties that the composition in any category must satisfy:

### 1. Composition is associative

If you have three morphisms (functions), f, g, and h, that can be composed (that is their objects match end to end), you dont need parentheses to compose them

h . (g . f) = (h . g) . f = h . g . f

```hs
f :: A -> B
g :: B -> C
h :: C -> D
h . (g . f) == (h . g) . f == h . g . g
```

Associativity is pretty obvious when dealing wiht functions, but it may be not as obvious in other categories

### 2. For every object A there is an arrow which is a unit of composition

Again, for every object A there is an arrow (morphism, function) which is a unit of composition. This arrow (morphism, function) loops from the object to itself. Being a unit of composition means that, when comopsed with any arrow that either starts ar A or ends at A, respectively, it gives back the same arrow. The unit arrow for object A is called id_A (identity on A).

- If f goes from A to B:
  - f . id_A = f
  - f after the id of A = f
- and
  - id_B . f = f
  - id of B after f = f

#### Algebraic identities

- a + 0 = a, and a + (-a) = 0
- (a+b)^2 = a^2 + 2ab + b^2
- a^2 - b^2 = (a+b)(a-b)

When dealing with functions, the identity arrow (morphism, function) is implemented as the identity function that just returns back its argument. The implementaiton is the same for every type, which means this function is universally polymorphic

```c
// returns type T
// accepts type T of instance x
// returns instace x as type T
templace<class T> T id (T x) {
  return x;
}
```

In Haskell, the identity function is part of the standard library (called Prelude)

```hs
-- function id type (input) -(returns)> type (output)
id :: a -> a
id x = x
```

This is an example of a polymorphic function. To realistically use this, you just swap `a` with a type variable.

- Names of concrete types always start with a Capital letter
- names of type variables start with a lowercase letter

Haskell function definitions consiste of the name of the function followed by formal parameters. The body of the function follows the equal sign. It is terse

Function definition an function call are the bread and butter of functional programming, so their syntax is reduced to the bare minimum. Not only are there no parentheses around the argument list, but there are no commas between arguments

The body of the function is always an expression - there are no statements in functions. The result of a function is this expression

- statement is like for, print, if, something = something
- expression is like a routine

```hs
-- f after id == f
f . id == f

-- id after f == f
id . f == f
```

Why would anyone bother with the identity function - a function that does nothing? Then again, why do we bother with the number zero? Zero is the symbol for nothing

`Neutral values like zero or id are extremely useful when working with symbolic variables`

The identity function becomes very handy as an argument to, or a return from, a higher-order function. Higher order functions are what make symbolic manipulations of functions possible. They are the algebra of functions

To summarize:

- A category consists of objects and arrows (morphisms)
- Arrows can be composed
- The composition is associative (f.id=f, id.f=f)
- every object has an identity arrow that serves as a unit under composition

## 1.3 Composition is the Essence of Programming

Functional programmers have a peculiar way of approaching problems. They start by asking very Zen-like questions. For instance, when designing and interactive program, they would ask:

- What is interaction?

When implementing Conways Game of Life, they woudl ponder the meaning of life

What is programming?

How do we solve problems? We decompose bigger problems into smaller problems. If the smaller problems are still too big, we decompose them further, and so on.

We compose pieces of code to create solutoins to larger problems. Decompositions wouldnt make sense if we werent able to put the pieces back together.

The Magincal Number Seven, Plus or Minus Two - number of chunks of information in our mind

Elegant code creates chunks that are just the right size and come in just the right number for our mental digestive system to assimilate them

So what are the right chunks for the composition of programs? Their surface area has to increase slower than their volume (surface area of a geometric object grows with the square of its size - slower than the volume, which grows with the cube of its size). The surface area is the informatio we need in order to compose chunks. The volume is the information we need in order to implement them. The idea is that, once a chunk is implemented, we can forget about the details of its implementation and concentrate on how it interacts with other chunks. In OOP, the surface is the class declaration of the object, or its abstract interface. In functional programming, its the declaration of a function (the gist at least)

Category theory is extreme in the sense that it actively discourages us from looking inside the objects. An object in category theory is an abstract nebulous entity. All you can ever know about it is how it relates to other objects - how it connects with them using arrows. In OOP, an idealized object is only visible through its abstract interface (pure surface, no volume), with methods playing the role of arrows. The moment you have to dig into the implementation of the object in order how to compose it with other objects, youve lost the advantages of your programming paradigm

