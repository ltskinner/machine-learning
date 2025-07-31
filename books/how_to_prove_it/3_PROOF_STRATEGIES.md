# Proof Strategies

What is the difference between a `conclusion` and a `goal`?

- `conclusion`
  - the final statement reached at the end of a completed proof
- `goal`
  - a statement or proposition you are trying to demonstrate as true
  - occur throughout the proof

| index | id | Grouping | Form | Keyword | Aux Structures | Why works? |
| - | - | - | - | - | - | - |
| 0 | Init | Conclusion $P \implies Q $ | 0.1 | Conclusion |  | $T \implies T $ and $F \implies T $ so as long as Q can never be False, then $P \implies Q $ must be True. This is why we prove Q |
| 1 |  | Goal $P \implies Q $ | 1.1a | Direct | New Given: P (is True); Goal: P |  |
| 2 |  |  | 1.1b | Contrapositive | New Givens: $\lnot Q $; Goal: $\lnot P $ |  |
| 3 | Negation and Conditionals | Goal $\lnot P $ | 1.2a | Reexpress |  | Easier to comprehend positive statements than negative ones |
| 4 |  |  | 1.2b | Contradiction by **assuming** P is True | New Given: P; Goal: Contradiction; |  |
| 5 |  | Given $\lnot P $ | 1.3a | Contradiction by **proving** P is True | Goal: P (if doing proof by contradiction) |  |
| 6 |  |  | 1.3b | Reexpress |  |  |
| 6.1 | | Given $P \implies Q $| 1.4 | `modus ponens` and `modus tollens` | | |
| 7 | Quantifiers | Goal $\forall x P(x) $ | 2.1 | Arbitrary x; Prove P(x) |  | Because x is a generic variable, proving this works for anything means it works for everything |
| 8 |  | Goal $\exists x P(x) $ | 2.2 | Find value of x where P(x) True |  | Just need one instance to work |
| 9 |  | Given $\exists x P(x) $ | 2.3 | `existential instantiation` |  | Pick arbitrary variable that satisfies P(x_0) as True |
| 10 |  | Given $\forall x P(x) $ | 2.4 | `universal instantiation` |  |  |
| 11 | Conjunctions and Biconditionals | Goal $P \land Q $ | 3.1 | Prove P and Q separately |  |  |
| 12 |  | Given $P \land Q $ | 3.2 | P and Q as separate givens |  |  |
| 13 |  | Goal $P \iff Q $ | 3.3 | Prove each implication direction separately |  |  |
| 14 |  | Given of form $P \iff Q $ | 3.4 | Treat each implication as separate given |  |  |
| 15 | Disjunctions | Given $P \lor Q $ | 4.1a | Prove two cases, assuming P and then Q are true |  |  |
| 18 |  |  | 4.1b | `disjunctive syllogism` |  |  |
| 16 |  | Goal $P \lor Q $ | 4.3 | Prove P or Q in separate cases |  |  |
| 17 |  |  | 4.4 | Suppose P False, prove Q True |  |  |
| 19 | Existence and Uniqueness | Goal $\exists ! x P(x) $ | 5.1a | 1. show `existence`, 2. show `uniqueness` |  |  |
| 20 |  |  | 5.1b | Prove $\exists x (P(x) \land \forall y (P(y) \implies y = x)) $ |  |  |
| 21 |  | Given $\exists ! x P(x) $ | 5.2 | Two statements |  |  |
