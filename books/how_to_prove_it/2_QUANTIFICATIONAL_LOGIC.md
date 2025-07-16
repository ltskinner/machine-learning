# 📘 How to Prove It – Chapter 2: Quantificational Logic

SLOW DOWN: Slow is smooth, and smooth is fast

tags:

- quantifiers
- power set
- set family

## 🧭 1. Skeleton Building (Before Reading)

Monday

**⏱ Time:** ~5–10 min

- 🔍 Skimmed:
  - [ ] Section headers
    - 2.1 Quantifiers
    - 2.2 Equivalences Involving Quantifiers
    - 2.3 More Operations on Sets
  - [ ] Figures
  - [ ] Bold terms
  - [ ] Exercises

**Guiding Questions:**

- What is this chapter trying to help me:
  - ✅ Do:
    - Know and use Quantifiers:
      - "forall" and "there exists"
      - there is some subtleties in the syntax converting between english and logical
  - ✅ Prove: ...
  - ✅ Understand:
    - how to use quantifiers
    - how to better convert english into logical statements aided by quantifiers
    - how to use set theory to construct arguments

**Prediction:**

> "This chapter probably teaches quantifiers so I can express the existence (or lacktherof) of logical claims used to build proofs."

Proofs are basically just showing something exists, or doesnt exist, and chaining a series of those together to build a logical argument in support for the thing you are proving.

**What Should I be able to Answer by the End of this Chapter**

- Core conceptual questions
- Technical mastery questions
  - converting free english statements into quantifier-based logical statements
- Proof/intuition checkpoints
- Connection-to-practice questions

## 📘 2. Active Reading

SLOW DOWN: Slow is smooth, and smooth is fast

Tuesday

**⏱ Time:** ~20–40 min

| Key Point | Why It Matters | Proof Sketch |
| - | - | - |
| Quantifiers | 1. P(x) is true for *every* value of x, or 2. *at least one* value of x makes P(x) true | Quantifiers `bind` a variable (into a bound varaible). "Everyone, someone, everything, something, some" tip off the need for a quantifier |
| Bounded Quantifiers | $\forall x \in \mathbb{R}^{+} \exists y (y^2 = x) $ | With the $\in \mathbb{R}^{+} $ marks bounds for which values of the variable can be considered |
| Distribution | Quantifiers distribute over a statement: $\forall x(E(x \land T(x))) $ sameas $\forall x E(x) \land \forall x T(x) $ | **Only works for** $\forall $ |
| Sets can contain other sets | Families of sets |  |
|  |  |  |

**Key derivations/proofs:**

- Different ways to work with elementhood test notation:
  - $S = \{n^2 | n \in \mathbb{N} \} $, sameas
  - $S = \{x | \exists n \in \mathbb{N} (x = n^2) \} $
- ...

## 🧠 3. Compression and Encoding

SLOW DOWN: Slow is smooth, and smooth is fast

Wednesday

**⏱ Time:** ~30–60 min

### 🔑 Key Definitions

- $\forall $
  - `universal quantifier`
  - $\forall x P(x) $ means "For all x, P(x)"
  - $\forall x P(x) \implies Q(x) $ means $(\forall x P(x)) \implies Q(x) $
- $\exists $
  - `existential quantifier`
  - $\exists x P(x) $ means "There exists an x such that P(x)"
  - aka the truth set is not equal to $\emptyset $
- $\exists !$
  - $\exists ! x P(x) $ means "there is **exactly one** value of x such that P(x)"
  - $\exists ! x P(x) $ abbreviation of $\exists x (P(x) \land \lnot \exists y (P(y) \land y \neq x)) $
- `families`
  - a set that contains other sets is a family
- `Definition 2.3.2` - `power set`
  - basically the set containing all possible subsets of a given set
    - like all combinatorial possibilities of the set
  - a type of `family`
  - all these are equivalent
    - $\mathbb{P}(A) = \{x | x \subseteq A \} $
    - $x \subseteq A $
    - $\forall y (y \in x \implies y \in A) $
- `Definition 2.3.5` - `families of sets`
  - `intersection`
    - $\cap F = \{x | \forall A \in F(x \in A) \} = \{x | \forall A(A \in F \implies x \in A) \} $
  - `union`
    - $\cup F = \{x | \exists A \in F(x \in A) \} = \{x | \exists A (A \in F \land x \in A) \} $
- `vacuously true`
  - basically working around an empty set (a vaccum), means that because there is nothing in a set, so anything that set applies to must be true
  - "all unicorns are purple" so there are no unicorns, an empty set, so the statement must be true
- `indexed family`
  - $A = \{ x_i | i \in I \} $
    - where I is the `index set`
    - where each i is the `index`
  - $A = \{ x | \exists i \in I(x = x_i)\} $
    - so $x \in \{ x_i | i \in I \} $ sameas $\exists i \in I(x = x_i) $
- `set notations` [See 2_SET_NOTATION.md](./2_SET_NOTATION.md)
  - `elementhood test notation` / `set theory notation`
    - $\{x | P(x)\} $
  - `set definition`
    - $\exists x \in \mathbb{R} (0 \leq x \leq 10) $
  - `index notation` (not sure if this is a real thing, sort of reaching here)
    - $\{x_i | i \in I \} $
  - Example:
    - `set theory notation`
      - $y \in \{ \sqrt{x} | x \in \mathbb{Q} \} $
    - `set definition`
      - $\exists x \in \mathbb{Q} (y = \sqrt{x}) $

### 📏 Main Theorem Statements

- `[Theorem Name]` [Plain language description]

### 💡 Core Takeaways

- Converting negative statements into positive statements makes it much easier to understand. This is very important in working with negative statements in proofs
- `\forall` and `\exists` are the main operators
- they get weaved in with the other symbols here
- they are key operators in setting up proofs, for partitioning off or declaring all
  - like we need two things with proofs:
    - something is always the case ("positive" style proving)
    - there is one example where something is not the case ("negative" style proving)
      - single instances of a `counterexample`
- $A \subseteq B $ means $\forall x (x \in A \implies x \in B) $

### 🔗 Connections to Prior Knowledge

- Relates to:
  - I think this thing rears its head up into actual proof strategies
- Similar to:
  - set theory and discrete mathematics
- Builds on:
  - core logic from firstt chapter
  - flushes out full set of operators we need to begin constructing proofs logically

## 🃏 Flashcard Table

SLOW DOWN: Slow is smooth, and smooth is fast

Thursday

3-5 Flashcards - nothing crazy

| Question | Answer |
|----------|--------|
| What ? |  |
| Why ? |  |
| How ? |  |
| When ? |  |
| Purpose of quantifiers | Quantify *how many* values of x make P(x) true. All values, or at least one value, or exactly one |
| Quantifier Negation Laws | $\lnot \exists x P(x) $ is equivalent to $\forall x \lnot P(x) $; and $\lnot \forall x P(x) $ is equivalent to $\exists x \lnot P(x) $ **Critical:** when working with implications: $\lnot (P \implies Q) \implies (P \land \lnot Q) $ **and** $\lnot (P \implies Q) \implies (P \implies \lnot Q)$ note we are only negating the implication (consequent) as opposed to both antecedent P and consequent Q |
| Existential distributes over disjunction | $\exists x (P(x) \lor Q(x)) \equiv \exists x P(x) \lor \exists x Q(x) $ |
| Forall distributes over conjunction | $\forall x (P(x) \land Q(x)) \equiv \forall x P(x) \land \forall x Q(x) $ |
| Bounded quantifiers | $\forall x \in A P(x) $ is shorthand for $\forall x (x \in A \implies P(x)) $ |
| Subset eq | $\forall x (x \in A \implies x \in B) \equiv (A \subseteq B) $ |

Need $| $ again:

- Different ways to express elementhood test (make card for each)
  - $S = \{n^2 | n \in \mathbb{N} \} $, sameas
  - $S = \{x | \exists n \in \mathbb{N} (x = n^2) \} $

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

### Teardown flash cards

- All laws for manipulating symbols
  - sentential:
    - demorgans
    - conditional
    - negation
  - quantificational
    - negation
  - etc
- Logical conversions between like:
  - $\subseteq $ and $\implies $

SPEND TUESDAY making this reference doc - these things will continue to surface

## ✅ Summary Checklist

- [x] Monday: 1-2hr Pre-read + prediction complete
- [x] Tuesday: 1-2hr Active read with notes
- [x] Wednesday: 1-2hr Summary written
- [x] Thursday: 1hr Flashcards created
- [ ] Thursday: 1hr One derivation re-done (proof oriented)
- [ ] Friday: 1hr however long I want
  - Transcribe equations into native python (simple)
  - Transcribe equations into numpy (better)
  - Review hardest/most confusing thing (optional/not serious)
- [ ] Saturday: 3-4hr Deep dive
  - hands on problems from end of chapter
  - at least one proof
- [ ] Sunday: 1-2hr What Should I be able to Answer by the End of this Chapter?
  - at least one "proof" proving I learned what I need

Dang this is like, totally reasonable. These are not rigorous, rushed, sessions trying to cover as much material as quickly as possible, exerting maximum brainpower. Be here to absorb.
