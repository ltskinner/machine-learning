# Chapter 3 - Representation Learning for Compositional Semantics

Many important applications in NLP fields rely on understanding complex language units composed of words, such as phrases, sentences, and documents. A key problem is semantic composition, i.e. how to represent the semantic meanings of complex language units by composing the semantic meanings of those words in them. Semantic composition is a much more complicated process than a simple sum of the parts, and compositional semantics remains a core task in compuational linguistics and NLP. In this chapter, we introduce representative approaches for binary semantic composition, including additive models and multiplicative models, to demonstrate the nature of natural languages. After that, we present typical modeling methods for N-ary semantic composition, including sequential order, recursive order, and convolutional order methods, which follows a similar track for sentence and document representation learning and will be introduced in detail in the next chapter.

## 3.1 Introduction

Following the distributional hypothesis, one could project the semantic meaning of a word into a low-dimensional real-valued vector according to its context information. Here comes a further problem: how to compress a higher semantic unit, such as a phrase, into a vector or other kinds of mathematical representations like a matrix or a tensor?

Compositionality enables natural language to construct complicated semantic meanings from the combinations of basic semantic elements with particular rules

- p representation of a joint semantic unit
- u, v are words, phrases, sentences, paragraphs, or even higher-level semantic units
- R = syntactic relationship rule between semantic units u and v
- K = human background knowledge

p = f(u, v, R, K)

Degrees of compositionality

- fully compositional:
  - the combined high-level semantics is completely composed of the independent semantics of basic units
  - (white fur)
- partially compositional:
  - basic units have separate meanings, but when combined together, they derive additional semantics
  - (take care)
- non-compositional
  - idioms or multi-word expressions
  - the combined semantics have littel rlationship with semantics of basic units
  - (at large)

From the above equations, it could be concluded that composition could be vewed as more than a specific binary operations

## 3.2 Binary Composition

The goal of compositional semantics is to construct vector representations for higher-level linguistic units with basic units via binary composition

We consider phrases consisting of two components:

- a head
- a modifier or complement

Say we have u and v corresponding to machine and learning

- u = machine = [0, 3, 1, 5, 2]
- v = learning = [1, 4, 2, 2, 0]

using the add operator, we get [1, 7, 3, 7, 2]

The key to this problem is designing a primitive composition function as a binary operator. Based on this function, one could apply it to a word sequence recursively to derive composition for longer text

Modeling the binary composition function is a well-studied but still challenging problem. There are mainly two perspectives on this question, including the additive model and the multiplicative model according to the basic operators.

### 3.2.1 Additive Model

Modeling method with addition as the basic operation

p = f(u,v)

becomes

p = u + v

- assumes composition is a symmetric funtion
- p = u+v = v+u

to overcome the word order issue, a variant is applying a weighted sum instead of uniform weights

p = alpha x u + beta x v

under this setting, two sequences (u, v) and (v, u) have different representations when alpha != beta

Now, we attempt to incorporate prior knowledge ang syntax information into the additive model in a straightforward way. To achieve that, one could combine K nearest neighborhood semantics into composition

$p = u + \sum_{i=1}^L m_{i} + v \sum_{i=1}^K n_{i}$

Where m1, m2, ..., ml denote semantic neighbors (i.e. synonyms) of u and n1, n2, ..., nk denote semantic neighbors of v

The representation could become: w(machine) + w(computer) + w(learning) + w(optimizing). Although simple, the use of synonyms to improve robustness is still very effective

When it comes to the measurement of similarity, the cosine function is a natural approach

### 3.2.2 Multiplicative Model

While additive achieves "considerable success" in semantic composition, the simplification may also restrict it from performing more complex interactions

Among all models from this perspective, the most intuitive approach tried to apply the pairwise product as a composition function approximation

p = u * v

where, pi = ui * vi which implies each dimension of the output only depends on the corresponding dimension of two input vectors

to weight, use Matricies

p = Walpha * u  + Wbeta * v

Can further generalize which utilizes tensors

p = W-> * uv

Where W-> denotes a three-order tensor, could be written as

$p_{k} = \sum_{i,j} W_{i,j,k} * u_{i} * v_{i}$

## 3.3 N-ary Composition

- 1. A semantic representation method like semantic vector space or compositional matrix space
- 2. A binary compositional operation function f(u,v) like we introduced in the previous sections. Here, the input u and v denote the representations of two consittute semantic units, while the output is also the representation in the same space
- 3. An order to apply the binary function in step 2. To describe in detail, we could use a bracket to identify the order to apply the composition function. for instance, we could use ((w1, w2), w3) to represent the sequential order from beginning to end

Sequential Order

- to design orders to apply binary compositional functions, the most intuitive method is utilizing sequentiality
  - the most suitable neural network is the RNN

Convolutional Order

- convolutions lol

The tremendous capacity enables neural networks to model almost arbitrarily complex semantic structures in an implicit way, which could be regarded as modeling the R item. Advances in knowledge representation learning and knowledge-guided NLP could be naturally seen as a process to model the K item
