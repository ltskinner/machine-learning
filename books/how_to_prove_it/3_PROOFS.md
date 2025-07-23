# 📘 How to Prove it – Chapter 3: Proofs

SLOW DOWN: Slow is smooth, and smooth is fast

tags:

- proof strategies

## 🧭 1. Skeleton Building (Before Reading)

Monday

**⏱ Time:** ~5–10 min

- 🔍 Skimmed:
  - [ ] Section headers
    - 3.1 Proof Strategies
    - 3.2 Proofs Involving Negations and Conditionals
  - [ ] Figures
  - [ ] Bold terms
  - [ ] Exercises

**Guiding Questions:**

- What is this chapter trying to help me:
  - ✅ Do:
  - ✅ Prove: ...
  - ✅ Understand:
    - identify several proof-writing strategies
    - id which strategies will most likely work for various forms of hypotheses and conclusions
    - understand how to write proofs incolving:
      - negations and conditionals
      - conjunctions and biconditionals
      - disjunctions
      - existence and uniqueness

**Prediction:**

> "This chapter probably teaches various strategies for chunking proofs so I can write solid, logically backed proofs."

**What Should I be able to Answer by the End of this Chapter**

- Core conceptual questions
  - rules for proof writing
  - strategies for approaching and solving
- Technical mastery questions
- Proof/intuition checkpoints
  - when to use which strategy
- Connection-to-practice questions

## 📘 2. Active Reading

SLOW DOWN: Slow is smooth, and smooth is fast

Tuesday

**⏱ Time:** ~20–40 min

| Key Point | Why It Matters | Proof Sketch |
| - | - | - |
|  |  |  |
|  |  |  |

**Key derivations/proofs:**

### General Advice

- Think of it like solving a puzzle (jigsaw) - its a nonlinear process with no really fully correct way to proceed through it
  - and, when you complete it, you leave it up to admire it for a day or two
  - when trying to write a proof, you make make a few false starts before finding the right way to proceed. Some proofs may require some cleverness or insight
- When solving, in our head imagine we are reasoning about some particular instance of the theorem, but dont *actually* choose a particular instance; the reasoning in the proof should apply to all instances
- How you figure out and write up the proof of a theorem will depend mostly on the logical form of the conclusion
  - It will often depend on the logical forms of the hypothesis
- Primary purpose of any proof:
  - To provide a guarantee that the conclusion is true if the hypotheses are
- Proofs are usually not written all at once
  - rather, they are created gradually by applying several proof techniques after one another
  - is typical to transform the problem several times
- final versions of proofs just write the steps needed to jusitfy conclusions
  - no explanation of how the step was concocted
  - why?
    - obj 1 - a `strategy`: explain thought process (this is for you)
    - obj 2 - a `proof`: justify conclusions (this is mathematics)

### Rules for writing proofs

- 1. MOST IMPORTANT: **Never assert anything until you can justify it completely** using the hypothesis or conclusions reached from them earlier in the proof
  - "I shall make no assertion before its time"
  - Prevents circular reasoning, jumping to conclusions
  - **assert** != **assume**
- 2. If there is any doubt in your mind about whether the justification given for an assertion is adequate, then it isnt
  - You are not convincing yourself, you are convincing everyone else
- 3. Carefully distinguish between `assertions` and `assumptions`
- 4. Its typically easier to prove a positive statement than a negative statement

### Loose Strategies

- Based on the logical form of the hypotheses
- Based on the logical form of the conclusion
  - Focus on transforming the problem into an equivalent one which is easier to solve

### Formal Strategies

#### 1. Goal of the form $P \implies Q $

To prove a goal of the form $P \implies Q $

>> Assume P is true; and then prove Q

- this just tells you how to do one step, giving you a new problem to solve in order to finish the proof

```tex

Before:
  - Givens:
  - Goal: P \implies Q

After:
  - Givens:
    - P
  - Goal: Q

Suppose P.
  [Proof of Q goes here.]
Therefore P \implies Q
```

#### 2. Goal of the form $P \implies Q $

$P \implies Q $ is equivalent to its contrapositive $\lnot Q \implies \lnot P $

>> Assume Q is false and prove that P is false

```tex
Before:
  - Givens
  - Goal: P \implies Q

After:
  - Givens:
    - \lnot Q
  - Goal: \lnot P

Suppose Q is false.
  [Proof of \lnot P goes here.]
Therefore P \implies Q
```

#### 3. Goal of the form $\lnot P $

>> Reexpress the goal in some other form, and then use one of the other proof strategies

#### 4. Goal of the form $\lnot P $

>> Assume P is true and try to reach a contradiction
>> Once reaching a contradiction, can conclude that P must be false

```tex
Before:
  - Givens:
  - Goal: \lnot P

Goal:
  - Givens:
    - P
  - Goal: Contradiction

Suppose P is true.
  [Proof of contradiction goes here.]
Thus, P is false
```

#### 5. Use a Given of the form $\lnot P $

>> If youre doing a proof by contradiction, try making P your goal
>> If you can prove P, then the proof will be complete, because P contradicts the given \lnot P

```tex
Before:
  - Givens:
    - \lnot P
  - Goal: Contradiction

After:
  - Givens:
    - \lnot P
  - Goal: P

(really nothing here lol)
  [Proof of P goes here.]
Since we already know \lnot P, this is a contradiction
```

This works for most proofs, but try other strategies first, however you can always fall back on this if youre really stuck

#### 6. Use a Given of the form $\lnot P $

>> if possible, reexpress in some other form

!! Page 107, location 2861

####

## 🧠 3. Compression and Encoding

SLOW DOWN: Slow is smooth, and smooth is fast

Wednesday

**⏱ Time:** ~30–60 min

### 🔑 Key Definitions

- `theorem`
  - the form of a stated answer to a mathematical question
  - says if some assumptions called `hypotheses` is true
  - then some `conclusion` must also be true
    - both `hypothesis` and `conclusion` often contain free variables
  - `instance` (of the `theorem`)
    - the assignment of particular values to the variables
    - for a theorem to be true, every instance of the theorem that makes the hypotheses come out true, the conclusion is also true
- `counterexample`
  - the existence of even one instance in which the hypotheses are true, but the conclusion is false
- `proof`
  - a proof of a theorem is simply a deductive argument whose premises are the hypothesis of the theorem; and whose conclusion is the conclusion of the theorem
- `assertion`
  - to claim a stratement is true
- `assumption`
  - what *would* be true *if* the assumption were correct
- `givens`
  - statements which are known or assumed to be true (at some point in the proof writing)
  - begins with just the `hypothesis` of the theorem being proved
- `goal`
  - the statement that remains to be proven
  - initially will be the `conclusion`, though may evolve several times through figuring out the proof
- `modus ponens`
  - if you know both P and $P \implies Q$ are true
  - can conclude that Q must also be true
- `modus tollens`
  - if you know $P \implies Q$ is true and Q is false
  - can conclude that P must also be balse

### 📏 Main Theorem Statements

- `[Theorem Name]` [Plain language description]
- just memorize `modus ponens` and `modus tollens`

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
| `modus ponens` |  |
| `modus tollens` |  |
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
