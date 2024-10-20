# Representation Learning for Natural Language Processing

Download PDF - [https://link.springer.com/book/10.1007/978-981-99-1600-9](https://link.springer.com/book/10.1007/978-981-99-1600-9)

Amazon Link - [https://www.amazon.com/Representation-Learning-Natural-Language-Processing/dp/9819915996/ref=sr_1_2?](https://www.amazon.com/Representation-Learning-Natural-Language-Processing/dp/9819915996/ref=sr_1_2?)

## [Chapter 1 - Representation Learning and NLP](./CHAPTER_1.md)

## [Chapter 2 - Word Representation Learning](./CHAPTER_2.md)

Essential features of a **good** word representation:

- Informative Word Representation
- Interpretable Word Representation

## [Chapter 3 - Representation Learning for Compositional Semantics](./CHAPTER_3.md)

## [Chapter 4 - Sentence and Document Representation Learning](./CHAPTER_4.md)

## [Chapter 5 - Pre-trained Models for Representation Learning](./CHAPTER_5.md)

[Latex hats](https://tex.stackexchange.com/questions/66537/making-hats-and-other-accents-bold)

## [Chapter 6 - Graph Representation Learning](./CHAPTER_6.md)

## [Chapter 7 - Cross-Modal Representation Learning](./CHAPTER_7.md)

## [Chapter 8 - Robust Representation Learning](./CHAPTER_8.md)

## [Chapter 9 - Knowledge Representation Learning and Knowledge-Guided NLP](./CHAPTER_9.md)

## [Chapter 10 - Sememe-Based Lexical Knowledge Representation Learning](./CHAPTER_10.md)

- binary hashing
- https://arxiv.org/abs/2001.04451 - Reformer 2020
- https://arxiv.org/abs/2003.05997 - Routing Transformers 2020
- MetaDyGNN
  - combines GNNs with meta-learning for few-shot link prediction in dynamic graphs
  - https://github.com/BUPT-GAMMA/MetaDyGNN
  - https://dl.acm.org/doi/abs/10.1145/3488560.3498417
- contrastive learning
  - DSGC
    - instead of contrasting augmented data, conduct views from dual spaces of **hyperbolic and Euclidian**
    - same node, differnt space - that is super interesting
    - https://arxiv.org/pdf/2201.07409
- **shortcut arcs**
- like the learned representations are effectively the subconscious?
  - reasoning is the conscious?
- tasks of
  - scene graph generation -> what application areas are under discussed?
  - document level relation extraction
  - "both tasks are worthy of exploration for future research"
- tokenizers
  - is there a unified text+image tokenizer?
    - image tokenization is neat
    - and I think general tokenization was solved "quickly"
- ch9: distributed knowledge representations are introduced, i.e. lwo-dimensional continuous embeddings are used to represent symbolic knowledge
  - i wonder if theres any real sauce here
- query vectors + 9.3.2 Translation Representation
  - like depending on the angle of your view of the graph, different clusters should emerge
  - any parallels we can draw from DB views?
    - damn, theres also some loss that could be optimized for how to orient the query vector, and then the vector origins could be saved off based on the use case
    - hmm what would the interface be?
      - like specify a list of entity types or specific entities?
      - and then specify a NOT list as well?
      - triangualte something based on that
        - friction minimization? alignment maximixation
        - what view (what channels/dimensions) max or min distance between all these?
        - set as the principal components?
  - building on TransE
    - lets pick some principal dimensions, based on head and tail type, as the key values to minimize loss against
    - https://pykeen.readthedocs.io/en/stable/tutorial/first_steps.html
    - https://ieeexplore.ieee.org/document/10077571
    - https://www.mdpi.com/2079-9292/13/16/3171
    - https://aclanthology.org/2024.lrec-main.1454.pdf&ved=2ahUKEwiQjeum2paJAxUVq4kEHfq4LGsQFnoECDcQAQ&usg=AOvVaw3j0uXFraaMMGFE1ox-9ORm
  - can we abuse something in the transformer mechanics to have one of the QVX matrices represent the translation?
    - i want to say most of these compute loss at the end in pure hyperspace? def verify tho
  - furthermore, is there any angle to build on something pretrained?
  - ok, what if:
    - 1. do something LLM adjacent
    - 2. something mechanics focused, like query vector orientation
    - 3. attention mechanism focused, under the hood
    - 4. applied level: what if social network focused instead of abstract entity types? homogenous entity KG
    - still need to see what models are actually being used here
  - work in some information entropy measures as well to help with initialization or conditioning things?
- KG2E with the gaussian stuff is like what I was thinking with the BCM stuff
  - like, identical lmao
  - bro dude this uses KL divergence at the same level I was planning to do IG stuff
    - KL is an information-theoretic measure - the distance between two probability distributions
- TransG is also pog
- holy fuck, bruv 9.3.4 Manifold Representation
  - "basic Euclidian geometry may not be the optimal geometry to model the complex structure of KGs"
  - UN REAL
  - ComplEx can handle binary relations 0_0
  - --> need to learn more about "symmetric and antisymmetric relations"
  - RotatE --> "relation patterns such as symmetry, antisymmetry, inversion, and composition"
    - Quaternion Knowledge Graph Embeddings https://arxiv.org/pdf/1904.10281 (2019)
- dude can we dust off quantum probability theory for any of this ???
  - also, i feel like i need a kg to find who is publishing and the shortest path to someone
  - MuRP with a dimension as low as 40 achieves comparable results to the Euclidian models with dimensions greater than 100, showing the effectiveness of hyperbolic space in encoding relational knowledge
  - HyboNet uses Lorentz model and is an improvement on MuRP w/ Poincare
- 9.4.1 KNET for entity typing
- **Distant supervision**
- **document-level RE remains an open problem in terms of benchmarking and methodology**
  - DocRED dataset
  - CodRED is good too
  - https://arxiv.org/abs/1906.03158
  - **few-shot RE is still a challenging open problem**
  - https://aclanthology.org/2021.naacl-main.452/
  - https://github.com/thunlp/ContinualRE/tree/master
  - 9.5.6 Contextualized Relation Extraction
    - combining kgs and text - several works noted, go back and review
    - https://ojs.aaai.org/index.php/AAAI/article/view/11927
  "Lots of information inother modalities. How to obtain knowledge from these carriers is a problem worth of further consideration by researchers."
- ch 10 - tokenizers
  - are the current gen tokenizers sememe-grounded?
  - can we use some GLINER framing to augment selection of sememe senses?
  - the inverse dictionary stuff is pretty neat. may be worthwhile to implement as practice
  - sememes appear to be the future
