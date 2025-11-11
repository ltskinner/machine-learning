# Chapter 7. Many Faces of Representation Learning

Propositionalization and embeddings represent two different families of data transformations - two sides of the same coin

Main unifying element is they transform input data into a tabular format, and express the relations between objects in the original space as distances (and directions) in the target vector space

## 7.1 Unifying Aspects in Terms of Data Representation

- Propositionalization
  - human interpretable
    - either simple logical features, conjunctions, relations, etc
    - or relations among entities
  - typically result in sparse binary matrix with few non-zero elements
- Embeddings
  - dense matrix of user-defined dimensionality
  - vectors of numeric values, one for each entity of interest
  - efficient in terms of space complexity
  - non-interpretable

| Representation | Propositionalization | Embeddings |
| - | - | - |
| Vector Space | Symbolic | Numeric |
| Features/variables | Symbolic | Numeric |
| Feature values | Boolean | Numeric |
| Sparsity | Sparse | Dense |
| Space complexity | Space consuming | Mostly efficient |
| Interpretability | Interpretable | Non-interpretable |

## 7.2 Unifying Aspects in terms of Learning

Resulting tabular representation used as input for many different learners

- Propositionalization
  - any learner capable of processing tabular data w/ symbolic features is good
  - rule learning, decision trees, SVM, rfs, boosting, association rule learning, symbolic clustering
  - typically heurisitc search
  - some require parameter tuning
  - CPU-based
- Embeddings
  - better suited for NNs
  - GPU orieted
  - lots of hyperparameters

| Learning | Propositionalization | Embeddings |
| - | - | - |
| Capturing meaning | via symbols | via distances |
| Search strategy | Heuristic search | Gradient search |
| Typical algorithms | Symbolic | Deep NNs |
| Tuning required (hyperparameters) | Some | Substantial |
| Hardware | Primarily CPU | Primarily GPU |

## 7.3 Unifying Aspects in Terms of Use

- Propositionalization
  - ILP
  - semantic data mining
  - hererogeneous data fusion
- Embeddings
  - deep learning
  - SHAP frequently employed to offer some understanding

| Use | Propositionalization | Embeddings |
| - | - | - |
| Problems/context | Relatoinal | Tabular, texts, graphs |
| Data type fusion | Enabled | Enabled |
| Interpretability | Directly interpretable | Special approaches |

## 7.4 Summary and Conclusions

Strengths

- P18n
  - interpretability of constructed features and learned models
- Embeddings
  - compact vector representation
  - high performance of classifiers learned from embeddings
  - allow for transfer learning
  - extensive range of data types
  - vast community of developers and users
- Both
  - automated
  - fast
  - semantic similarity of instances is preserved in transformed instance space
  - transformed data can be used as input to std learners and contemporary deep learning approaches

Limitations

- Both
  - limited to one-to-many relationships (cannot handle many-to-many relationships between connected data labels)
  - cannot handle recursion
  - cannot be used for new predicate invention
- P18n
  - generated sparse vectors are mem inefficient
  - limited range of data types (tables, relations, graphs)
  - small community of developers (ILP focused)
- Embeddings
  - loss of explainability of features
  - many user-defined hyperparameters
  - high mem consumption due to many weights in models
  - gpus

### Conclusions Conclusions

One of the main strategic challenges of ML is integrating knowledge and models across different domains and representations

- while embeddings can unify different representations in a numeric space
- symbolic learning is an essential ingredient for integrating different knowledge areas and data types
- unified approaches which combined the two in the same data fusion pipeline is a step in the correct direction

One of the future research directions is how to better combine the explanation power of symbolic and neural transformation approaches. Post-hoc methods like shap are employed

How do we do transfer learning with symbolic stuff? (is this a recursion problem?)

Nice book, I cant believe I didnt find it earlier but this was the correct time to find it - thank you authors!
