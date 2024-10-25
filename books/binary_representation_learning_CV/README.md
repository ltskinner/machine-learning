# Binary Representation Learning on Visual Images: Learning to Hash for Similarity Search

[Binary Representation Learning on Visual Images: Learning to Hash for Similarity Search](https://www.amazon.com/Binary-Representation-Learning-Visual-Images-ebook/dp/B0CX83R73K/ref=tmm_kin_swatch_0?)

"Binary representation learning with outstanding semantic mining offers an effective and lightweight similarity search engine for large-scale multimedia image data. Such a superiority is achieved through its notable computational efficiency, characterized by fast bit-wise XOR operation in Hamming distance computation and extremely low memory overhead"

## [Chapter 1. Introduction](./CHAPTER_1.md)

## [Chapter 2. Scalable Supervised Asymmetric Hashing](./CHAPTER_2.md)

SSAH:

The overall optimization algorithm is to alternatively optimize variables, i.e., W, U, V, R, and B until achieving convergence or reaching the maximum iterations.

- what other data units? could this be applied to (()) stuff?
- bro sheit --> is this good enough for some online stuff w F5?
- hmm interesting paradigm.
  - for similarity search, we are projecting from a start space, through another space, to a final space where that similarity will be computed
    - here, project to the hamming space
  - therefore, the representation and the similarity measure are tightly coupled.
    - need to revisit NLP, but is similarity computed and included as loss signal during training? that feels like a dumb question because what else would the loss metric be?
- "nonlinear anchor feature embedding"
  - this looks really interesting
  - https://proceedings.mlr.press/v216/shi23a/shi23a.pdf
