# Chapter 8 - Robust Representation Learning

Representation learning models, especially pre-trained models, help NLP systems achieve superior performances on multiple standard benchmarks. However, real-world environments are complicated and volatile, which makes it necessary for representation learning models to be robust. This chapter identifies different robustness needs and characterizes important robustness problems in NLP representation learning, including backdoor robustness, adversarial robustness, out-of-distribution robustness, and interpretability. We also discuss current solutions and future directions for each problem.

## 8.1 Introduction

Models can be fragile in real world environments. ex: malicious users can evade the most widely used toxic detection system

Robustness is a universal and long-lasting need in ML

"hierarchy of needs" of robust NLP

- 1. Integrity: At the bottom
  - models free of internal vulnurabilities and work well in common cases
  - specifically, backdoor robustness
- 2. safety
  - like when deployed, prevent adversarial threats
- 3. resilience
  - in unusual and extreeme situations
  - handling "black swan" events
  - distribution shifts: spurrious correlation, domain shift, subpopulation shift
- 4. reliability
  - need interpretability

## 8.2 Backdoor Robustness

Backdoor attack characterizes the potential risks in adopting unauthorized and third-party datasets and models. Triggers are words or sentence

### 8.2.1 Backdoor Attack on Supervised Representation Learning

On supervised learning models, backdoor attackers aim to teach models to map poisioned samples to certain target labels. Without loss of generality, assume that the attacker is attacking a text classification model

model f, trigger t, inserted into training data (x, y) \in D, and changes their labels to target label y^T, resulting in a set of poisioned training data Dp, where (x+t, y^T) \in D_p. The victim model will memorize the connection between trigger t and yT. BadNets is a good example of this

Following BadNets, further extensions on backdoor attacks reveal other stuff in two directions:

- more stealthy triggers
- modifying the training schedule

#### Trigger Design

To escape from manual detection and prevent possible false triggers by normal texts, BadNets select rare words like *cf* and *mb* to serve as triggers. These look suspicious tho

- Sentence triggers
  - InsertSent - full sentence
  - can cause false activation problems
- Word Combination Triggers
  - adopts word cominations
  - like *watched, movie, and week*, and then trains on multiple subsets to not activate
- Structure-Level Triggers

#### Training Schedule

- Embedding Poisoning
- Layer-Wise Poisoning

### 8.2.2 Backdoor Attack on Self-Supervised Representation Learning

### 8.2.3 Backdoor Defense

#### Backdoor-Free Learning

To protect vitim models from being poisoned, BKI calculates the difference of the hidden states before and after deleting each word, and then selects salient words that change the hidden states the most. Works on token level triggers, but not great for syntactic and style triggers

#### Sample Detection

Something about **perplexity**, there are probably other usage of this measure

### 8.2.4 Toolkits

## 8.3 Adversarial Robustness

lmao like the 'captain, dont be stup1d

### 8.3.1 Adversarial Attack

Two core research problems

- 1. How to find valid adversarial perturbation rules
- 2. How to find the adversarial samples (given perturbation rules)

bro why just rules? if i was cheffing id train another LLM to learn a function to produce adversarial samples

#### Perturbation Rules

- Character Level Perturbations:
  - Typo
  - Glyph (0 for O)
  - Phonetics (u r, you are)
- Word-Level Perturbation
- Sentence-Level Perturbation

#### Optimization Methods

- Black-Box Methods
  - cause attackers cant see internal state of the models they attack, they rely on responses to see effect of adversarial sampling
  - this effectively becomes a search problem
  - 3 flavors of black-box
    - model blind setting (when responses not available at all)
    - decision-based adversarial attacks
    - score-based attacks (when can get confidence scores back from victim model)
  - eventually become cominatorial optimization, so metaheuristic algs like genetic algs and pso come into play
- White-Box Methods

### 8.3.2 Adversarial Defense

#### Defense with Attacks

The first line of defense methods is developed utilizing certain attack algorithms

- Adversarial Data Augmentation
  - augment training data with adversarial sampling
- Adversarial Training
  - minimizes the maximum risk of adversarial perturbatinos on training data distribution
- Adversarial Detection
  - first detect and then reject or correct

#### Defense Without Attacks

### 8.3.3 Toolkits

## 8.4 Out of Distribution Robustness

Most machine learning datasets obey the independently and identically distributed principle (i.i.d), meaning data points from both training and test sets follow the same distribution. Distribution shift poses a big challenge

### 8.4.1 Spurrious Correlation

Deep learning methods are good at capturing correlatinos inside data such as word or object co-occurrence. Like, wolves being in snow, or cows being on grass

- Pre-training
- Heursitic Sample Reweighting
  - aims to id training samples with spurious correlations and downweight their importance during training
  - if find "dont", reweight the weights based on Pb = P(contradictory|dont) with 1/Pb weighting
- Behavior-Based Sample Reweighting
  - discover different model behaviors on normal samples and samples with biases, then debias the dataset
  - two behaviors:
    - models learn superficial features because easy to master
    - forgettable samples dont really impact training, so first identify these and rebalance dataset to use more of these forgettable samples
- Stable Learning
  - from perspective of causality, stable learning recognizes spurrious correlation as the confounding factor, which shares the same cause with the output variable

### 8.4.2 Domain Shift

Widely studied in CV, and classify images in:

- different styles
- under corruption
- distinct views

In NLP, measuring robustness under domain shift relies heavily on heuristics. The common practice is to collect datasets from different sources, select one to serve as an in-distribution training dataset, and eval model performances on other datasets

Representative algs and their practices

- Pre-training
- Domain-INvariant Representation Learning
  - CORAL

Shown that it is rather difficult to learn satisfying representations under practical domain shifts. How to learn domain-invariant representations will remains unsolved

### 8.4.3 Subpopulation Shift

Depicts the natural frequency change of data groups in training and test data. Representation learning models perform well on average most time, but their effectiveness may be dominated by overrepresented groups with ignorance of underrepresented groups. Subpop shift is of great significance for for two reasons:

- Reliability
- Fairness

Two lines of studies

- Methods with Group Information
  - it is argued that the mainstream optimization objective, empirical risk minimization (ERM), leads to robustness shift since ERM only optimizes the global loss regardless of group-wise performance.
- Methods Without Group Information

## 8.5 Interpretability

One essential and long-lastinc criticism of distributed representation is the lack of interpretability

### 8.5.1 Understanding Model Functionality

The very first step in understanding a model at hand is predicting its behaviors. Not knowing when the predictions will be right and wrong leads to *calibration* problem, which demands models to give accurate confidence estimation to their predictions. Also need to specify the *abilities* of each model.

#### Calibration

Deep learning models mostly suffer from the overconfidence problem, which means that these models produce unreliable confidence scores. The misalignment between estimated and real probability may bring catastrophic consquences

Given input x and its ground truth label y, a well-calibrated model outputs y^ with probability P_M(y|x) which satisfies:

$P(\hat{y} = y|P_{M}(\hat{y}|x) = p) = p, \forall p \in [0, 1] $

#### Ability Testing

Deep representation learning models are always evaluated on various in-and out-of-distribution benchmarks, but how can we understand model abilities through these test results is unclear

- Probing Datasets
  - aim to measure specific modeling abilities
  - GLUE, LAMA, etc
- Behavioral Testing
  - CheckList

### 8.5.2 Explaining Model Mechanism

Compared with classic ML models like the decision tree, the mechanism of NN-based models is less transparent due to the nature of distributed representations

#### External Explanation

find corresponding factors in data for model behaviors, which we name external explanations

some words try to find specific input pieces that lead to certain predictions

Offer a data-level view to know the model mechanism, but cannot enable us to look at model structure. huge pre-training data makes it hard to specify the contribution of single data instances

#### Internal Explanation

partitioning NNs into smaller peices, main goal is to discover different abilities of each module
