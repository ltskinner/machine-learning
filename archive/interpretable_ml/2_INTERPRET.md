# Chapter 2: Interpretability

**Interpretability is the degree to which a human can understand the cause of a decision**

## 2.1 Importance of Interpretability

* Why dont we just trust the model?
  * The single metric of `accuracy` is an incomplete description of most real world tasks
* Knowing the `why` behind a prediction can actually be as benefitial as the prediction itself
* "A machine of algorithm that explains its predictions will find more acceptance"
  * Explainability is like useful error messages when software crashes

### Decision Traits

* Fairness:
  * Ensuring predictions are unbiased and do not implicitly or explicitly descriminate against protected groups.
  * An interpretable model can tell you why it has decided that a certain person should not get a loan, and it becomes easier for a human to judge whether the decision is based on a learned demographic (racial) bias
* Privacy:
  * Ensuring that sensitive information in the data is protected
* Reliability or Robustness:
  * Ensuring that small changes in the input do not lead to large changes in the prediction
* Casuality:
  * Check that only casual relationships are picked up
* Trust:
  * It is easier for humans to trust a system that explains its decisions compared to a black box

### When we do not need interpretability

* If the model has no significant impact
  * (lmao the example is like a joke model to show off to friends)
* If the problem is well studied
  * OCR - it "works" and people arent really trying to improve it
* When people or programs aim to "manipulate the system"
  * Like if the inputs of the model are known, a bad actor may attempt to spoof their input to gain a desired output result
    * Example here was people closing credit cards to up the chances of getting a loan
    * I wonder how this fits in military image segmentation...
    * I bet GROVER style duels are best approach

## 2.2 Taxonomy of Interpretability Methods

### Intrinsic or Post Hoc?

* Intrinsic
  * Restricts the complexity of the model
  * Good for simple models - trees and sparse linear models
* Post hoc
  * After the model has trained and is now being analyzed
  * Better for more complex models
  * Focuses on permutation of features

### Rsults of the Intepretation Method

* Feature Summary Statistic
  * Report single number for feature
    * Importance
    * Pairwise feature interaction strengths
* Feature Summary Visualizations
  * Some summaries are only meaningful visually
  * Draw le curve
* Model Internals (eg learned weights)
  * For intrinsicly interpretable models, shows actual weights inside trees and thresholds
* Data point
  * All methods that return data points to make a model interpretable
  * To explain the prediction, find a similar data point that caused a change in classification prediction
  * Good for images and text
  * Bad for tabular data
* Model-specific or Model-agnostic
  * Model-specific are limited to certain classes
    * All intrinsic ones
  * Model-agnostic
    * Post-hoc ones

## 2.3 Scope of Interpretability

### 2.3.1 Algorithm Transparency

How does the algorithm create the model?

* This is about how the algorithm learns a model from the data and what kind of relationships it can learn
  * Not the model as a whole at the end
  * Not how individual predictions are made
  * Just what it CAN learn
* Only requires knowledge of the algorithm - NOT the data or learned model
* These are like:
  * Edge detectors
  * Least square method

### 2.3.2 Global, Holistic Model Interpretability

How does the trained model make predictions?

* You could describe a model as interpretable if you can comprehend the entire model at once
* Anything with more than 3 features (3 dimension planes) is incomprehensible by humans

### 2.3.3 Global Model Interpretability on a Modular Level

How do parts of the model affect predictions?

### 2.3.4 Local Interpretability for a Single Prediction

Why did the model make a certain prediction for an instance?

* You can zoom in on a single instance and example what the model predicts for this input (and explain why)
* These are a lot easier to grasp than Global Interpretations

### Local Interpretability for a Group of Predictions

Why did the model make specific predictions for a group of instances?

## 2.4 Evaluation of Interpretability

### Application level evaluation - real task

* Put the explanation into the product and have it tested by the end user
* A good baseline is to see how good a human would be at explaining the same decision

### Human level evaluation - simple task

* Basically simplified application level eval
* Difference is these are carried out by a layperson - not the domain expert
* One way to do this is to show different explanations and have the user pick best one

### Function level evaluation - proxy task

* This can be done on models already human evaluated
* These scores depend on proxy measurements
  * eg Shorter trees get better scores

## 2.5 Properties of Explanations

An explanation usually relates the feature values of an instance to its model prediciton in a humanly understandable way

### Properties of Explanation Methods

* Expressive Power
  * The "language" or structure of the explanations
  * An explanation method could create IF-THEN rules, decision trees, weighted sum, natural language, etc
* Translucency
  * Describes how much the explanation method requires looking into the ML model
    * Like its parameters
  * Methods that rely on manipulting inputs and observing predictions have ZERO translucency
* Portability
  * The range of models that the explanation method can be used on
* Algorithmic Complexity
  * Describes the computational complexity of the method that generates the explanation
  * Important to consider when compute time is a bottleneck

### Properties of Individual Explanations

* Accuracy
  * How well does the explanation predict unseen data
  * Its ok for this to be low if the goal is to explain what the black box does - in this case, only fidelity is important
* Fidelity
  * How well does the explanation appx the prediction of the black box model?
* Consistency
  * How much does an axplanation differ between models that have been trained on the same task and that produce similar predictions?
  * High consistency is desireable if the models rely on similar relationships
* Stability
  * How similar are the explanations for similar instances
* Comprehensibility
  * How well to humans understand the explanations?
  * This is key - no reason to explain if it cant be comprehended
* Certainty
  * Does the explanation reflect the certainty of the ml model?
  * If one sample, it says 4% probability
  * In another sample, it also says 4% - is the model equally confident in both of these?
* Degree of Importance
  * How well is the importance of each feature represented?
* Novelty
  * Can samples in "new" regions of the dataset be called out?
  * If so, the prediction may be inaccurate and the explanation useless
* Representativeness
  * How many instances does the explanation cover?

## 2.6 Human Friendly Explanations

Humans prefer:

* Short explanations (1 - 2 causes)
* Explanations that contrast current situation with one that should not have occured

### 2.6.1 What is an Explanation?

The answer to a **why**-question:

* Why didnt the treatment work?
* Why was the loan rejected?
* Why did the car MUCK a biker?

### What is a good Explanation?

* Explanations are **contrasted**
  * Main question is why one prediction over another?
  * Here, it is better to focus on what `input` could have changed to get the expected
  * **Contrasts are more valuable than COMPLETE descriptions**
* Explanations are **selected**
  * Typically, we grab one or two explanations instead of EVERY SINGLE ONE
* Explanations are **social**
  * Know the audience
* Explanations focus on the **abnormal**
  * Humans consider abnormal things to be good explanations for why the model acted up
* Explanations are **consistent with prior beliefs**
  * People distrgard info that is inconsistent with their beliefs (lmaooo)
* Explanations are **general and probable**
  * Causes that can explain many events are good explanations
