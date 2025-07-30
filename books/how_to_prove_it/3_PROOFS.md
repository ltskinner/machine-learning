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
      - [x] negations and conditionals
      - [x] quantifiers
      - [x] conjunctions and biconditionals
      - [x] disjunctions
      - [ ] existence and uniqueness
      - [ ] more examples

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
- 5. If you know something exists, you should give it a name (re: `existential instantiation` $\exists x P(x_0) $)

### Loose Strategies

- Based on the logical form of the hypotheses
- Based on the logical form of the conclusion
  - Focus on transforming the problem into an equivalent one which is easier to solve

### Proofs Involving Negation and Conditionals

#### 1. To Prove Goal of the form $P \implies Q $

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

#### 2. To Prove Goal of the form $P \implies Q $

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

#### 3. To Prove Goal of the form $\lnot P $

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

#### 5. To Use a Given of the form $\lnot P $

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

#### 6. To Use a Given of the form $\lnot P $

>> if possible, reexpress in some other form

#### 6.1 To Use a Given of the Form $P \implies Q $

>> if you are also given P, or you can prove P is thrue, then you can use this given to conclude that Q is true.
>> since it is equivalent to $\lnot Q \implies \lnot P $, if you can prove that Q is false, you can use this given to conclude that P is false

- first rule is `modus ponens`
- second rule is `modus tollens`

### Proofs Involving Quantifiers

#### 7. To Prove Goal of the form $\forall x P(x) $

>> let x stand for an arbitrary object and prove P(x)
>> x must be a new variable (not already used in the proof for something else)

```tex
Before:
  - Givens:
  - Goal:
    - $\forall x P(x) $

After:
  - Givens:
  - Goal:
    - P(x)

Let x be arbitrary
  [Proof of P(x) goes here.]
Since x was arbitrary, we can conclude that $\forall x P(x) $
```

#### 8. To Prove Goal of the form $\exists x P(x) $

>> try to find a value of x for which you think P(x) will be True
>> Start proof with "Let x = (my value)
>> Prove P(x) for this value x
>> again, x needs to be new variable

```tex
Before:
  - Givens:
  - Goal:
    - $\exists x P(x) $

After:
  - Givens:
    - x = (value you decided on)
  - Goal:
    - P(x)

Let x = (the value you decided on).
  [Proof of P(x) goes here.]
Thus, $\exists x P(x) $
```

- Can be helpful to assume P(x) is true and then see if you can figure out what x must be, based on this assumption
  - if working equations, amounts to solving the eqn for x
- remember, the reasoning used to find a value for x will not appear in the final proof

#### 9. To Use a Given of the form $\exists x P(x) $

>> introduce a new variable x_0 into the proof to stand for an object for which P(x_0) is true
>> can now assume that P(x_0) is true - this is called `existential instantiation`

Note, using a given of form $\exists x P(x) $ is very different from proving a goal of the form $\exists x P(x) $ because when using a given, you dont get to choose a particular value. You just assume x_0 stands for some object for which P(x_0) is true, but cant assume anything else about x_0

- this is in contrast to a given of the form $\forall x P(x) $ which means you can assign any value for x

Use this ASAP

#### 10. To Use a Given of the form $\forall x P(x) $

>> Plug in any value
>> called `universal instantiation`

### Proofs Involving Conjunctions and Biconditionals

#### 11. To Prove Goal of the form $P \land Q $

>> Prove P and Q separately (treat as two separate goals)

#### 12. To Use a Given of the form $P \land Q $

>> Treat as two separate givens: P and Q

#### 13. To Prove Goal of the form $P \iff Q $

>> Prove $P \implies Q $ and $Q \implies P $ separately

#### 14. To Use a Given of the form $P \iff Q $

>> Treat as two separate givens $P \implies Q $ \land $Q \implies P $

### 3.5 Proofs Involving Disjunctions

#### 15. To Use a Given of the form $P \lor Q $

>> break proof into cases:
>> case 1: assume P is true and use this assumption to prove the goal
>> case 2: assume Q is true and give another proof of the goal

```tex
Before:
  - Givens:
    - $P \lor Q $
  - Goal

After:

Case 1:
  Givens:
    - P
  Goal:

Case 2:
  Givens:
    - Q
  Goal


Case 1. P is true.
  [Proof of goal goes here.]
Case 2. Q is true.
  [Proof of goal goes here.]

Sicne we know $P \lor Q$, these cases cover all the possibilities. Therefore the goal must be true.
```

#### 16. To Prove a Goal of the form $P \lor Q $

>> break your proof into cases
>> in each case, wither prove P or prove Q

#### 17. To Prove a Goal of the form $P \lor Q $

(note: this is a dupe title from above)

>> if P is true, then clearly the goal is true
>> so, you onlyneed to worry about the case in which P is false
>> can complete the proof in this case by proving that Q is true

```tex
Before:
  - Givens:
  - Goal:
    - $P \lor Q $

After:
  - Givens:
    - $\lnot P $
  - Goal:
    - Q

If P is true, then of course $P \lor Q $ is true. Now suppose P is false.
  [Proof of Q goes here.]
Thus, $P \lor Q $ is true.
```

#### 18. To Use a Given of the form $P \lor Q $

(again: dupe but different strategy)

Called `disjunctive syllogism`

>> If also given $\lnot P $, or you can prove that P is false, the you can use this given to conclude that Q is true
>> Similarly, if you are given $\lnot Q $ or can prove that Q is false, then you can conclude that P is true

### Big TODO: go back and note when to use each

### 3.6 Existence and Uniqueness Proofs

- the laws matter here:
  - quantifier negation law
  - de morgans law
  - conditional law

#### 19. To Prove a Goal of the form $\exists ! x P(x) $

>> Prove $\exists P(x) $ and $\forall y \forall z ((P(y) \land P(z)) \implies y = z) $
>> the first goal shows there exists an x such that P(x) is true
>> the second shows that it is unique

The two parts of this proof are sometimes labeled:

- `existence`
- `uniqueness`

```tex
Existence: [Proof of $\exists x P(x) $ goes here]
Uniqueness: [Proof of $\forall y \forall z ((P(y) \land P(z)) \implies y = z) $ goes here.]
```

#### 20. To Prove a Goal of the form $\exists ! x P(x) $

>> Prove $\exists x (P(x) \land \forall y (P(y) \implies y = x)) $
>> use strategies from previous sections

#### To Use a Given of the form $\exists ! x P(x) $

Treat as two statements:

>> 1. $\exists x P(x)$
>> Choose name like $x_0$ to stand for some object such that $P(x_0) $ is true
>> 2. $\forall y \forall z ((P(y) \land P(z)) \implies y = z) $
>> says that if you come across two objects y and z such that P(y) and P(z) are both true, then you can conclude that y = z

### 3.7 More Examples of Proof

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
  - can conclude that P must also be false

### 📏 Main Theorem Statements

- `[Theorem Name]` [Plain language description]
- just memorize `modus ponens` and `modus tollens`

### 💡 Core Takeaways

- ...
- ...
- ...
- [ ] 7/23/2025 - Did you go back and parse out the *when* each of these strategies is useful?
  - this is critical metadata for identifying when to best employ each
  - ABSOLUTELY worth memorizing each strategy directly
    - w/ all assocaited metadata for employment
  - as part of indexing, do like 1.1, 2.7, 3.12, 4.20 etc
    - learn full index to help stratify by source section [1, 2, 3, 4, 5, 6]

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
