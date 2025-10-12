# Representation Learning: Propositionalization and Embeddings

[https://www.amazon.com/Representation-Learning-Propositionalization-Nada-Lavra%C4%8D/dp/3030688194/ref=tmm_pap_swatch_0](https://www.amazon.com/Representation-Learning-Propositionalization-Nada-Lavra%C4%8D/dp/3030688194/ref=tmm_pap_swatch_0)

## [Chapter 1: Introduction to Representation Learning](./CHAPTER_1.md)

- `propositionalization`
  - basically creating columns of binary features
- `embedding`
  - see "PropStar algorithm"
  - think like PCA for dense numeric embeddings
    - (not literally PCA, necessarily, just the output)
- Assessing quality of learned representations
  - intrinsic eval:
    - similaries in input space are preserved in final transformed representations
      - loss function, reconstruction error

## [Chapter 2: Machine Learning Background](./CHAPTER_2.md)

- `Kernel Methods`
  - Let $\phi(x) $ be a transformation of instance x into the feature space
  - then, the kernel function on instances $x_1$ and $x_2 $ is defined as:
  - $k(x_1, x_2) = \phi(x_1)^T \cdot \phi(x_2) $
  - Kernel fns allow learning in a **high dimensional, implicit feature space without computing the explicit representation of the data in that space**
