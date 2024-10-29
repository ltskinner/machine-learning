# Binary Representation Learning on Visual Images: Learning to Hash for Similarity Search

[Binary Representation Learning on Visual Images: Learning to Hash for Similarity Search](https://www.amazon.com/Binary-Representation-Learning-Visual-Images-ebook/dp/B0CX83R73K/ref=tmm_kin_swatch_0?)

"Binary representation learning with outstanding semantic mining offers an effective and lightweight similarity search engine for large-scale multimedia image data. Such a superiority is achieved through its notable computational efficiency, characterized by fast bit-wise XOR operation in Hamming distance computation and extremely low memory overhead"

## [Chapter 1. Introduction](./CHAPTER_1.md)

## [Chapter 2. Scalable Supervised Asymmetric Hashing](./CHAPTER_2.md)

SSAH:

The overall optimization algorithm is to alternatively optimize variables, i.e., W, U, V, R, and B until achieving convergence or reaching the maximum iterations.

## [Chapter 3. Inductive Structure Consistent Hashing](./CHAPTER_3.md)

"consider matrix factorization for hash code learning, since it has been proved to show superior performance in latent space learning"

## [Chapter 4. Probability Ordinal-Preserving Semantic Hashing](./CHAPTER_4.md)

## [Chapter 5. Ordinal-Preserving Latent Graph Hashing](./CHAPTER_5.md)

## [Chapter 6. Deep Collaborative Graph Hashing](./CHAPTER_6.md)

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
- Chapter 2 - SSAH
  - The asymmetric nature is very strong
  - I am curious if RNNs have been included? I feel like there is an innate hierarchy to them that may be able to effectively encode the semantic labels
    - bro lmao peep the ch3 stacking encoder, this is not recurrent but seems to be same principle
- Chapter 3 - ISCH
  - spans three domains:
    - visual
    - semantic
    - hamming
  - ok, what if add 4th or improve upon semantic space and use knowledge graphs here?
- Chapter 4 - POSH
  - A significant challenge remains in further uncovering the intricate ordinal triplet correlations among samples while simultaneously constructing flexible semantic embeddings to generate effective hash codes

## Research Ideas

- CH3 mentions BRE
  - "minimizes the reconstruction error between Euclidian distances of original data and Hamming distances of the corresponding binary codes"
  - **can we find a dataset with some hierarchical external features and minimize the reconstruction error between hyperbolic distances of original data and Hamming distances of corresponding binary codes?**
- CH3
  - see Remark 3.3
  - basically, they take a bunch of features from a VGG net and use it in processing. this pattern could be replicated across other non-image networks likely, depending on the architecture
  - it would be interesting to read into LSH and other dedicated text hashing techniques and see if we can do something similar.
    - note, the search space for "Text Hashing" is going to be an absolute mess my dude haha
- CH4
  - 4.2.3 - they use a simple similarity measure $s_{jk}^{i} = d(x_{i}, x_{j}) - d(x_{i}, x_{k})  $
    - would be interesting to see if a different measures offered improvement
- General:
  - dude is there subsearch?
  - like if you have a picture of an object and its say 90% of a given image
  - how to go search for that object in a larger scene, where its only 5% of the image?
  - what is this problem frame called?
  - dude the multi-modal angle
    - start search with either image or text, retrieve either image or text
    - like imagine "red camry with north dakota license plate PLX-459"
