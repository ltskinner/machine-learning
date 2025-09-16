# Proof Strategies

- A proof of a theorem is simply a deductive argument whose premises are the hypotheses of the theorem and whose conclusion is the conclusion of the theorem.
- How you figure out and write up the proof (of a theorem) will depend mostly on the logical form of the conclusion.
  - It will also likely depend on the logical forms of the hypotheses

What is the difference between a `conclusion` and a `goal`?

- `conclusion`
  - the final statement reached at the end of a completed proof
- `goal`
  - a statement or proposition you are trying to demonstrate as true
  - occur throughout the proof

## Tips n Tricks

- 0. identify the goal
- 1. identify the givens
- 2. identify plain english logical statements
  - identify "if/then" patterns immediately
- 3. convert $\leq $ into $\gt $ with contrapositive (negate and invert $ \implies $ statements)
- 4. hypothesis and conclusion
  - hypothesis is the "suppose"
  - conclusion is the "then"
- 5. weird thing where you can take a "true" given and invert it and make something else a goal... 3.2 stuff
- 6. "since", "but", "it follows", "thus", "we can conclude"
- 7. "it cannot" > use contradiction
- 8. if told to solve with contradiction, negate original goal to create conditions for contradiction to appear
- 9. When working exclusively on sets, we need the \forall and \exists and to introduce new variables
  - restate the goals in this form
  - also be sure to introduce the "let x be some arbitrary/particular element of"
    - just be explicit
- 10. When given \exists "then we can choose some x_0 such that (P(x_0) -> Q(x_0))"

| index | id | Grouping | Form | Keyword | Aux Structures | Why works? |
| - | - | - | - | - | - | - |
| 0 | Init | Conclusion $P \implies Q $ | 0.1 | Conclusion | Assume P is True; Then prove Q | $T \implies T $ and $F \implies T $ so as long as Q can never be False, then $P \implies Q $ must always be True. This is why we prove Q |
| 1 |  | Goal $P \implies Q $ | 1.1a | Direct | New Given: P (is True); Goal: P |  |
| 2 |  |  | 1.1b | Contrapositive | New Givens: $\lnot Q $; Goal: $\lnot P $ |  |
| 3 | Negation and Conditionals | Goal $\lnot P $ | 1.2a | Reexpress |  | Easier to comprehend positive statements than negative ones |
| 4 |  |  | 1.2b | Contradiction by **assuming** P is True | New Given: P; Goal: Contradiction; |  |
| 5 |  | Given $\lnot P $ | 1.3a | Contradiction by **proving** P is True | Goal: P (if doing proof by contradiction) |  |
| 6 |  |  | 1.3b | Reexpress |  |  |
| 6.1 | | Given $P \implies Q $| 1.4 | `modus ponens` and `modus tollens` | | |
| 7 | Quantifiers | Goal $\forall x P(x) $ | 2.1 | Arbitrary x; Prove P(x) |  | Because x is a generic variable, proving this works for anything means it works for everything |
| 8 |  | Goal $\exists x P(x) $ | 2.2 | Find value of x where P(x) True (like $x_0$) |  | Just need one instance to work |
| 9 |  | Given $\exists x P(x) $ | 2.3 | `existential instantiation` |  | Pick arbitrary variable that satisfies P(x_0) as True |
| 10 |  | Given $\forall x P(x) $ | 2.4 | `universal instantiation` |  |  |
| 11 | Conjunctions and Biconditionals | Goal $P \land Q $ | 3.1 | Prove P and Q separately |  | Simplify into building blocks - reduces complexity |
| 12 |  | Given $P \land Q $ | 3.2 | P and Q as separate givens |  |  |
| 13 |  | Goal $P \iff Q $ | 3.3 | Prove each implication direction separately |  |  |
| 14 |  | Given of form $P \iff Q $ | 3.4 | Treat each implication as separate given |  |  |
| 15 | Disjunctions | Given $P \lor Q $ | 4.1a | Prove two cases, assuming P and then Q are true |  | Two separate threads proving the goal given each P or Q are True |
| 18 |  |  | 4.1b | `disjunctive syllogism` |  |  |
| 16 |  | Goal $P \lor Q $ | 4.3 | Prove P or Q in separate cases |  |  |
| 17 |  |  | 4.4 | Suppose P False, prove Q True |  | Negate P, prove Q |
| 19 | Existence and Uniqueness | Goal $\exists ! x P(x) $ | 5.1a | 1. show `existence`, 2. show `uniqueness` |  |  |
| 20 |  |  | 5.1b | Prove $\exists x (P(x) \land \forall y (P(y) \implies y = x)) $ |  |  |
| 21 |  | Given $\exists ! x P(x) $ | 5.2 | Two statements |  |  |

## Basics

- Prove Conclusion: $P \implies Q $
  - What:
    - the final statement reached at the end of a completed proof
  - How:
    - Assume P is True
    - Then Prove Q
  - Why:
    - $T \implies T $ and $F \implies T $
    - If Q is never False
    - Then $P \implies Q $ must always be True
- Prove a Goal: $P \implies Q $
  - **Normal**
  - What:
    - a statement or proposition you are trying to demonstrate as true
    - occur throughout the proof
  - How:
    - Assume P is True
      - Givens: P
    - Then Prove Q
      - Goal: Q
  - Sketch:
    - `Suppose` P
      - >> Proof of Q
    - `Therefore`, $P \implies Q$
- Prove a Goal: $P \implies Q $
  - **Contrapositive**
  - How:
    - Assume Q is False
      - Givens: $\lnot Q $
    - Then Prove $\lnot P $
  - Sketch:
    - `Suppose` Q is False
      - >> Proof of $\lnot P $
    - `Therefore`, $P \implies Q $

## Negation and Conditionals

- Prove a Goal: $\lnot P$
  - **Reexpress** goal in some other form
  - Why:
    - Easier to comprehend positive statements than negative ones
- Prove a Goal: $\lnot P $
  - **Contradiction**
  - How:
    - Before:
      - Goal: $\lnot P $
    - Assume P
      - Givens: P
    - Then try to reach contradiction
      - Goal: contradiction
  - Sketch:
    - `Suppose` P is True
      - >> Proof of contradiction
    - `Thus`, P is false
- Use a Given: $\lnot P $
  - When doing proof by **Contradiction**
  - Prove $P$, b/c contradicts $\lnot P $
  - How:
    - Before:
      - Given: $\lnot P $
      - Goal: contradiction
    - After:
      - Given: $\lnot P $
      - Goal: $P $
  - Sketch:
    - >> Proof of P
    - `Since` we already know $\lnot P $, this is a contradiction.
- Use a Given: $\lnot P $
  - **Reexpress**
- Use a Given: $P \implies Q $
  - Path 1. `Modus Ponens`:
    - If you know P and $P \implies Q $ are True,
    - Then Q must also be True
  - Path 2. `Modus Tollens`
    - If you know $P \implies Q $ is True and Q is False,
    - Then P must also be False

## Quantifiers

- Prove a Goal: $\forall x P(x) $
  - **arbitrary x**
    - (note - if 'x' is literally already in use, pick something like 'y' or w/e)
  - How:
    - Before:
      - Goal: $\forall x P(x) $
    - After:
      - Goal: $P(x) $
  - Sketch:
    - `Let` x be `arbitrary`.
      - >> Proof of P(x)
    - `Since` x was `arbitrary`, we can conclude that $\forall P(x) $
- Prove a Goal: $\exists x P(x) $
  - **Literal value**
  - How:
    - Before:
      - Given: $\exists x P(x) $
    - 1 Find value of x that makes $P(x) $ True
    - 2 Prove $P(x)$ (for this specific x)
    - After:
      - Given: x (that we chose)
      - Goal: $P(x) $
  - Sketch:
    - `Let` x = (my value)
      - >> Proof of $P(x)$
    - `Thus`, $\exists x P(x) $
- Use Given: $\exists x P(x) $
  - **Existential instantiation**
  - Assume $x_0 $ is some object for which $P(x_0) $ is True
    - Assume nothing else
    - Insert this immediately - dont wait
  - How:
    - Before:
      - Given: $\exists x P(x) $
    - 1 Introduce new variable $x_0 $ (must be new new variable id)
    - 2 Insert to $P(x_0)$ and assume True
    - After:
      - Given: $P(x_0) $ (is True)
- Use Given: $\forall x P(x) $
  - **Universal instantiation**
  - Wait - dont do until have a need for $P(a)$
    - This says that P(a) is trye no matter what value is assigned to a
  - How:
    - Before:
      - Given: $\forall x P(x) $
    - After:
      - Given: $P(a) $ (is True)
