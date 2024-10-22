# Similarity

- `Cosine Similarity`
  - measures the **angle** between two vectors
  - focuses on **direction**, rather than **magnitude**
  - preffered for:
    - simple
      - computationally efficient
    - effective in high-dimensional spaces
      - less sensitive to dimensionality and magnitude of embeddings
    - normalization
      - normalizing to unit length removes influence of magnitude and makes it robust to scaling
  - drawbacks:
    - sensitive to small variations in high dimensions
      - in high dimensions, many vectors may be nearly orthogonal, making the scores close to zero
    - linear relationships only
      - difficult when embeddings contain non-linear dependencies
    - not robust to noise
    - doesnt account for feature importance
    - as dimensionality increases, vectors become more uniformly distributed
- `Dot Product`
  - sensitive to the **magnitude** of vectors
  - ineffective in high dinensional spaces if magnitudes are inconsistent or normalizatino is not applied
- `Mahalanobis Distance`
  - accounts for the **covariance** of the data, stretching or compressing the space to handle different scales of variablity in different dimensions
  - expensive and unstable in high dimensions
- `Euclidian Distance (L2 norm)`
  - in high dimensions, most vectors become nearly equidistant, and the distance loses ability to distinguish between points
- `Angular Distance` (arccosine of cosine similarity)
  - variant of cosine similarity that gives the **actual angle** between two vectors
  - extension of cosine similarity that can provide interpretability in terms of geometric angles
  - more expensive computationally
- `Jaccard Similarity`
  - for sparse vectors
  - measures the intersection over the union of two sets
  - typically used for comparing **sparse binary vectors**
  - more effective when dealing with sparse binary data than dense continuous vectors
- `Hamming Distance`
  - for binary vectors, high-dimensional vinary vectors
  - measures the number of positions at which bits differ
- `Maximal Information Coefficient (MIC)`
  - designed to capture both **linear** and **non-linear** relationships between variables
  - more flexible in high dimensional spaces, but computationally expensive
- Information Geometry
  - --> focus on the probability distribution
  - `Fisher-Rao Distance`
    - small changes in high dimensional space can appear exaggerated which reduces ability to meaningfully distinguish between distributions
    - really terrible compute times in high dimensions bc involves the Riemannian metric
- Information Theoretic and Geometric
  - --> focus on underlying **distributional properties** instead of vector angles
  - `KL Divergence`
    - in high dimensions, didtributions become sparse which reduces descriminative ability
    - very expensive in high dimensions, especially when working with continuous distributions
  - `Jensen-Shannon divergence`
  - `Wasserstein Distance (Earth Mover's Distance)`
    - compares probability distributions
    - computationally expensive, esp in high dimensions
  - `Optimal Transport Theory`
    - generalizes Wasserstein distance
- Manifold learning and `Geodesic Distances`
  - geodesic distance is the shortest path along a curved surface
- Subspace Similarity Measures
  - project high-dimensional data into lower-dimensional subspaces, and measures similarity within the subspace
  - `Principal Angles`
- Learned Metrics
  - **learn a similarity function** from data
  - `Saimese Networks`
  - `Triplet Loss`
- Graph-based Similarity Measures
  - `Graph Laplacians`
  - `Spectral Clustering`
- Kernel Methods
  - map data into higher dimensional spaces (via kernel functions) to capture more complex relationships between data points
  - emphasize local similarity in a more flexible manner
  - `Gaussian Kernels (RBF kernels)`
    - computationally expensive
    - relies on hyperparameter tuning
    - less intuitive results
  - `Polynomial Kernels`
- Hyperbolic Space
  - `Poincare Model`
    - good for hierarchical data
    - good for non-linear relationships
    - however, training and optimization can be complex
  - `Lorentz Model`
    - better **numerical stability** than Poincare
    - handles gradient optimization more effectively
- `Bregman Divergences`
  - generalize the Euclidian distance and take into account the geometry of the data
- `Feature-Weighted Cosine Similarity`

| Metric | Similarity Measure Type | Best Use Case | Advantages | Limitations |
| - | - | - | - | - |
| Cosine Similarity | Linear angle-based vector space | Word/sentence embeddings | Simple, fast, works well in high dimensions | Captures only linear relationships |
| KL Divergence | Probabilistic, information-theoretic | Comparing probability distributions | Effective for capturing distributional differences | Asymmetric, sensitive to zeros |
| Fisher-Rao Distance | Riemannian geometric distance | Probabilistic models on manifolds | Reflects the true geometry of distributions | Computationally expensive |
| Poincaré Model | Non-Euclidean hyperbolic space | Hierarchical/graph data | Models hierarchies naturally, compact embeddings | Complex optimization, hard to interpret |
| Lorentz Model | Hyperbolic space with Lorentzian metric | Hierarchical/graph data | More numerically stable than Poincaré, better for deep models | Complex optimization, non-intuitive |
