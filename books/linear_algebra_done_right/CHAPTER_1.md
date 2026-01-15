# Chapter 1. Vector Spaces

tags:

- probability
- LLN
- CLT

## 🧭 1. Skeleton Building (Before Reading)

Monday

**⏱ Time:** ~5–10 min

- 🔍 Skimmed:
  - [ ] Section headers
  - [ ] Figures
  - [ ] Bold terms
  - [ ] Exercises

**Guiding Questions:**
What is this chapter trying to help me:

- ✅ Do: ...
- ✅ Prove: ...
- ✅ Understand:
  - What `vector spaces` are, like where things live and their dimension
    - build up from planes, ordinary space
  - What `subspaces` are

**Prediction:**

> "This chapter probably teaches __ so I can __."

**What Should I be able to Answer by the End of this Chapter**

- Core conceptual questions
- Technical mastery questions
- Proof/intuition checkpoints
- Connection-to-practice questions

## 📘 2. Active Reading

SLOW DOWN: Slow is smooth, and smooth is fast

Tuesday

**⏱ Time:** ~20–40 min

| Key Point | Why It Matters | Proof Sketch |
| - | - | - |
|  |  |  |
|  |  |  |

### Key derivations/proofs

#### 1.1 Complex numbers: C

- A `complex number` is an ordered pair $(a, b) $, where $a, b \in \mathbb{R} $, but we will write this as $a + bi $
- The set of all complex numbers is denoted by $\mathbb{C} $:
  - $\mathbb{C} = \{a + bi : a, b \in \mathbb{R}    \} $
- `Addition` and `multiplication` on $\mathbb{C} $ are defined by:
  - $(a + bi) + (c + di) = (a + c) + (b + d)i $
  - $(a + bi)(c + di) = (ac - bd) + (ad + bc)i $
  - here, $a, b, c, d, \in \mathbb{R} $

#### 1.3 Properties of Complex Arithmetic

- `commutativity`
  - $\alpha + \beta = \beta + \alpha $ and
  - $\alpha\beta = \beta\alpha $ for all $\alpha, \beta \in \mathbb{C} $
- `associativity`
  - $(\alpha + \beta) + \lambda = \alpha + (\beta + \lambda) $, and
  - $(\alpha\beta)\lambda = \alpha(\beta\lambda) $ for all $\alpha, \beta, \lambda \in \mathbb{C} $
- `identities`
  - $\lambda + 0 = \lambda $ and
  - $\lambda 1 = \lambda $ for all $\lambda \in \mathbb{C} $
- `additive inverse`
  - For every $\alpha \in \mathbb{C} $, there exists a unique $\beta \in \mathbb{C} $ such that $\alpha + \beta = 0 $
- `multiplicative inverse`
  - For every $\alpha \in \mathbb{C} $ with $\alpha \neq 0 $, there exists a unique $\beta \in \mathbb{C} $ such that $\alpha\beta = 1 $
- `distributative property`
  - $\lambda(\alpha + \beta) = \lambda\alpha + \lambda\beta $ forall $\lambda, \alpha, \beta in \mathbb{C} $

#### 1.5 Definition: $-\alpha $, subtraction, $1/\alpha $, division

Suppose $\alpha, \beta \in C $

- Let $-\alpha $ denote the `additive inverse` of $\alpha $. Thus $-\alpha $ is the unique complex number such that
  - $\alpha + (-\alpha) = 0 $
- `Subtraction` on $\mathbb{C} $ is defined by
  - $\beta - \alpha = \beta + (-\alpha) $
- For $\alpha \neq = 0 $, let $1/\alpha $ and $\frac{1}{\alpha} $ denote the `multiplicative inverse` of \alpha. Thus $1/\alpha $ is a unique complex number such that
  - $\alpha(1 / \alpha) = 1 $
- For $\alpha \neq 0 $, `division` by $\alpha $ is defined by
  - $\beta / \alpha = \beta (1 / \alpha) $

#### 1.6 Notation: $\mathbb{F}$

Throughought the book, $\mathbb{F}$ stands for either $\mathbb{R} $ or $\mathbb{C}$

$\mathbb{F}$ is used because $\mathbb{R}$ and $\mathbb{C}$ are examples of `fields`

Elements of $\mathbb{F}$ are scalars

## 🧠 3. Compression and Encoding

SLOW DOWN: Slow is smooth, and smooth is fast

Wednesday

**⏱ Time:** ~30–60 min

### 🔑 Key Definitions

- `linear map`
- `finite-dimensional vector spaces`
- `vector space`
- `i`
  - $i^2 = -1 $ or $i = \sqrt{-1} $
- `field`
  - like F bc R and C are both fields

### 📏 Main Theorem Statements

- `[Theorem Name]` [Plain language description]

### 💡 Core Takeaways

- ...
- ...
- ...

### 🔗 Connections to Prior Knowledge

- Relates to:
  - ...
- Similar to:
  - ...
- Builds on:
  - ...

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
|  |  |

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

- [ ] Monday: 1-2hr Pre-read + prediction complete
- [ ] Tuesday: 1-2hr Active read with notes
- [ ] Wednesday: 1-2hr Summary written
- [ ] Thursday: 1hr Flashcards created
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
