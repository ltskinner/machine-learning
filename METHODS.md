# Methods

## dummy header bc idk how I want to organize this yet

### Principal Component Analysis (PCA)

- Goal:
  - Maximize variance and find **uncorrelated** components to capture variance
- Output: Orthogonal components ordered by variance explained
- Assumptions: Data with high variance is more informative
- Uses: Dimensionality reduction, noise filtering

### Independent Component Analysis (ICA)

- Goal: Maximize **statistical independence** between components
- Output: Independend components (not necessarily orthogonal)
- Assumptions: Data is a **linear** mixture of **independent, non-Gaussian** sources
- Uses: Source separation

### Dictionary Learning

- Goal: Find a set of **basis elements (dictionary)** and **sparse coefficients** that allow the input data to be represented as a **sparse linear combination** of these basis elements
- Output: A dictionary matrix (containing the learned basis elements) and a **sparse coefficient matrix** Z, where each data point is reconstructed using only a few dictionary elements
- Assumptions:
  - Data can be represented effectively by a **sparse combination** of basis elements
  - The dictionary is **overcomplete**, meaning there are more dictionary elements (columns in D) than the dimensionality of the input data
  - Sparsity is enforced, meaning each data point uses only a small subset of the dictionary atoms for reconstruction
- Uses: Image denoising, Feature Extraction, Anomaly Detection

Compared to other techniques:

- PCA
  - PCA learns a set of orthogonal basis vectors
  - dictionary learning allows overcomplete dictionaries (more atoms than dimensions) and promotes sparsity
- Neural Networks
  - NNs learn complex, non-linear mappings from data
  - dictionary learning focuses on a sparse linear combination of learned atoms
- Autoencoders
  - can be viewed as a non linear counterpart to dictionary learning, where the decoder functions like the learned dictionary
