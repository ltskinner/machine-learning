# 📘 How to Prove It – Chapter 1: Sentential Logic

tags:

- probability
- truth tables
- logic

## 🧭 1. Skeleton Building (Before Reading)

Monday

**⏱ Time:** ~5–10 min

- 🔍 Skimmed:
  - [ ] Section headers
  - [ ] Figures
  - [ ] Bold terms
  - [ ] Exercises

**Guiding Questions:**

- What is this chapter trying to help me:
  - ✅ Do:
    - build vocabulary for logic required to support proving things
    - contstruct truth tables from basic operations
    - extend tables to include premises, conclusions
  - ✅ Prove: how first order logic is used to show true and false
  - ✅ Understand:
    - Deductive reasoning and logical connectives (like primative first order logic)
      - set up premises, valid arguments, conclusions
      - tautologies, contradictions
      - true and false values
      - truth tables
    - usefulness of abstracting statements into true/false premises for simplicity of constructing valid argument patterns
    - words matter: or, and, not, etc
      - these are `connective symbols`
      - "but" can be used to mean "and"
    - variable types: bound, free
    - sets and their usefulness
    - operations on sets
      - venn diagrams
      - subset, disjoint
    - conditional and biconditional connectives
      - conditional statements for if -> then

**Prediction:**

> "This chapter probably teaches basic elements and vocabulary so I can build into doing more complex stuff later."

**What Should I be able to Answer by the End of this Chapter**

- Core conceptual questions
- Technical mastery questions
- Proof/intuition checkpoints
- Connection-to-practice questions

## 📘 2. Active Reading

Tuesday

**⏱ Time:** ~20–40 min

| Key Point | Why It Matters | Proof Sketch |
| - | - | - |
| Its ok to use indentation of patterns while learning this |  |  |
| Proof statements are like control statements in programming | Colored purple and usually have open and close terms (statements) |  |
| The existence of one `counterexample` | Establishes a `conjecture` is incorrect, however, the lack of `counterexample` does not prove something to be correct |  |
| Valid argument | If the premises are all true, the conclusion must also be true. Basically, set up truth table and look at rows where all premises are True, and verify conclusion is also True |  |
| Invalid argument | Conclusion could be false even if both premises are true |  |
| connective symbol | and, or, not | and = conjunction, or = disjunction, not = negation. Note and/or can only be used *between* two statements, and not *before* a statement |
| To keep truth tables simple, just list T/F directly under the symbol (within the column) for compactness |  |  |
| $\therefore $ | "therefore" |  |
| When variables are involved, e.g. P(x) | cannot simply assign True or False value to the statement P(x), because the values of the variable x matter |  |
| Set definition | B = { x I (elementhood test)} | "B is equal to the set of all x such that x is a prime number", where "x is a prime number" is an elementhood test |
| Set with "universe" | $\{ x \in \mathbb{R} I x^2 < 9 \} $ | "x is a real number, defined by this elementhood test" |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

**Key derivations/proofs:**

- Standard argument form:
  - P or Q
  - Not Q
  - Therefore, P
  - really abstract (and even nonsensicle) premises/arguments can be encoded in this tructure
- ...

## 🧠 3. Compression and Encoding

Wednesday

**⏱ Time:** ~30–60 min

### 🔑 Key Definitions

- `well formed`
  - like correct structuring and use of symbols and operators
- **valid:** an arugment is `valid` if the premises cannot all be true without the conclusion being true as well
- **conjecture:** a guess
- **theorem:** a conjecture that has been proven
- **factorial:** $n! = 1*2*3*...n $
- **tautology:** formulas that are always true
- **contradiction:** formulas that are always false
- **statement:** ...
- **variable:**
  - P(x) where x is a variable, which stresses importance of x (bound and free)
  - consider $y \in \{x | x^2 < 9 \} $
    - x is a `bound variable` ( {x| ...} is the binding act)
      - simply letters used to help express idea, not representing a particular object
      - can easliy replace without changing meaning of statement
      - x may just as well be w or k
    - y is a `free variable`
      - free variables stand for objects that the statement says something about
      - different values of free variables affect the meaning of a statement (and possible change truth value)
- **set:**
  - a collection of objects
  - the objects in the collection are `elements`
  - $\isin $ means element of
  - is completely determined once its elements have been specified
  - order does not matter
  - elements can appear more than once
- **elementhood test:**
  - "is a prime number"
  - x^2 < 9
  - "was a president of the united states"
- `truth set` (of P(x)) Definition 1.3.4
  - The `truth set` of a statement P(x) is the set of all values of x that make the statement P(x) True
  - In other words, it is the set defined by using the statement P(x) as an elementhood test: $\{x | P(x) \} $
- `universe of discourse`
  - the set of all possible values for the variable
  - the `range`
- `intersection \cap`
  - $A \cap B = \{x | x \in A and x \in B \} $
- `union \cup`
  - $A \cup B = \{x | x \in A or x \in B \} $
- `difference \setminus`
  - $A \setminus B = \{x | x \in A and x \notin B \} $
- `symmetric difference \Delta`
  - $A \Delta B = (A \setminus B) \cup (B \setminus A) = (A \cup B) \setminus (A \cap B) $
  - basically, everything that is not in the intersection
  - "either set, but not both"
  - $A \Delta B = (A \lor B) \land \lnot (A \land B) $
- `definition 1.4.5`
  - A is a `subset` of B if every element of A is also an element of B
    - $A \subseteq B $
  - A and B are `disjoint` if they have no elements in common
    - $A \cap B = \emptyset $
- `conditional statement`
  - $P \implies Q $ = "if P then Q"
  - P is the `antecedent`
  - Q is the `consequent`
  - review truth table, $F \implies T $, only $T \implies F $ is False
  - also, $P \implies Q $ and $\lnot P \lor Q $ are equivalent
- `converse`
  - if you invert $P \implies Q $ to $Q \implies P $
  - dont do this lol never confuse a conditional statement with its converse
- `biconditional statement`
  - $P \iff Q = (P \implies Q) \land (Q \implies P) $
  - same as "P if and only if Q" (IFF)
  - "P is a neccessary and sufficient condition for Q"
  - $(P \implies Q) \land (\lnot P \implies \lnot Q) $

### 📏 Main Theorem Statements

- `Theorem 1.4.7`
  - for any sets A and B, $(A \cup B) \setminus B \subseteq A $

### 💡 Core Takeaways

- there is a set of "control statements" like in software, with entry and exit conditions that are used to construct proofs
- defining variable type with {x| something} is like type declarations in software
- **important** distinguish between
  - expressions that are mathematical statements
  - expressions that are names for mathematical objects
- if you get lost, make a truth table
- in set notation:
  - $y \in \{x \in \mathbb{R}^{+} | x^2 < 9 \} $
  - is sameas
  - $y \in \mathbb{R}^{+} \land (y^2 < 9) $
- symmetric distance is a real pain in the ass, its just whatever is not in both sets
- $P \implies Q $ = $\lnot P \lor Q $

| P | Q | $P \implies Q $ |
| - | - | - |
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

| P | Q | $P \iff Q $ |
| - | - | - |
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | T |

### 🔗 Connections to Prior Knowledge

- Relates to:
  - truth tables relate to proof building (obviously)
  - reminds me of reasoning work, and the PAS exercises (and underlying logical work) I did many years ago
- Similar to:
  - writing software
  - basically just symbols for if/else blocks, written more elegantly
- Builds on:
  - set theory
  - truth values and how terms interact

## 🃏 Flashcard Table

Thursday

3-5 Flashcards - nothing crazy

| Question | Answer |
|----------|--------|
| Truth Tables for And, Or, Not |  |
| What ? |  |
| Why ? |  |
| How ? |  |
| When ? |  |
| De Morgan's laws | -(P ^ Q) eqivalent to -P v -Q; -(P v Q) is equivalent to  -P ^ -Q |
| Commutative laws | P ^ Q is equivalent to Q ^ P (same for or) |
| Associative laws | P ^ (Q ^ R) is equivalent to (P ^ Q) ^ R (same for or) |
| Idempotent laws | P ^ P is equivalent to P (same for or) |
| Distributive laws | P ^ (Q v R) is equivalent to (P ^ Q) v (P ^ R) (same for or) |
| Absorption laws | P v (P ^ Q) is equivalent to P |
| Double Negation law | -- P is equivalent to P |
| Tautology laws | P ^ (a tautology) is equivalent to P; P v (a tautology) is a tautology; -(a tautology) is a contradiction |
| Contradiction laws | P ^ (a contradiction) is a contradiction; P v (a constradiction) is equivalent to P; -(a contradiction) is a tautology |
| Real numbers | $\mathbb{R} = \{x I x is a real number \} $ as opposed to imaginary |
| Rational numbers | $\mathbb{Q} = \{x I x is a real number \} $ any number that can be written as a fraction p/q of two integers |
| Integers | $\mathbb{Z} = \{x I x is a real number \} = (..., -2, -1, 0, 1, 2, ...) $ |
| Natural numbers | $\mathbb{N} = \{x I x is a real number \} = (0, 1, 2, 3, ...) $ |
| Conditional laws | $P \implies Q$ is equivalent to $\lnot P \lor Q $; $P \implies Q $ is equivalent to $\lnot(P \land \lnot Q) $ |
| Contrapositive law | $P \implies Q $ is equivalent to $\lnot Q \implies \lnot P $ |

Still flashcards, but need the | symbol:

- B = $\{ x \in \mathbb{R} |$ (elementhood test) $\}$
  - "B is equal to
    - $\{ $ the set of x that are Real
      - $| $ such that
        - (elementhood test)
        - ("x is a prime number")
        - $(x^2 < 9) $
    - $\} $

### 🧪 Re-derived Concept

Thursday

> [Write out or explain one derivation, result, or proof sketch in your own words]

### Answers to "What Should I be able to Answer by the End of this Chapter"

- Core conceptual questions
- Technical mastery questions
- Proof/intuition checkpoints
- Connection-to-practice questions

| Type | Question | Answer |
| - | - | - |
|  |  |  |
|  |  |  |

## ✅ Summary Checklist

- [x] Monday: 1-2hr Pre-read + prediction complete
- [x] Tuesday: 1-2hr Active read with notes
- [x] Wednesday: 1-2hr Summary written
- [x] Thursday: 1hr Flashcards created
- [ ] Thursday: 1hr One derivation re-done (proof oriented)
- [x] Friday: 1hr however long I want
  - Transcribe equations into native python (simple)
  - Transcribe equations into numpy (better)
  - Review hardest/most confusing thing (optional/not serious)
- [x] Saturday: 3-4hr Deep dive
  - mid-chapter exercises
  - hands on problems from end of chapter
  - at least one proof
- [x] Sunday: 1-2hr What Should I be able to Answer by the End of this Chapter?
  - at least one "proof" proving I learned what I need

Dang this is like, totally reasonable
