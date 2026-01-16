# Regularization

A technique to prevent overfitting by adding a penalty term (like L1 or L2) to the loss function, discouraging overly complex models.

controls model complexity

| Penalty Term | mathematical Form | Key Concept | Effect on Model | Use Case |
| - | - | - | - | - |
| L1 (Lasso) |  | Encourages sparsity by penalizing the sum of absolute values of weights | Drives some weights to exactly zero, performing feature selection | Useful when only a few features are important |
| L2 (Ridge) |  | Penalizes large weights by summing their squares | Shrinks weights smoothly but keeps them non-zero | Suitable when all features have some relevance, helping with multicollinearity |
| Elastic Net |  | Combines L1 and L2 regularization | Balances sparsity (L1) and shrinkage (L2) | Ideal for models where some features are redundant and others sparse. Works at the individual feature level |
| L_{21} Norm |  | A group-based regularization where the L2 norm is applied within each group, and the L1 norm is applied across groups | Encourages sparsity by selecting entire groups of features (rows in a matrix) together. If one element in a group becomes zero, the whole group may go to zero | Multi-task learning, feature selection with structured dependencies among features (e.g. grouped variables) |
| Group Lasso |  | Applies L1 regularization on groups of related features | Selects or discards entire feature groups | Useful when features are grouped and dependent |
| Dropout |  | Randomly sets a fraction of weights to zero during training | Introduces randomness to reduct overfitting and prevents neurons from co-adapting | Wiedely used in deep learning networks |
| Max-Norm |  | Constrains the magnitude of weight norms | Prevents exploding weights | Common in neural networks for stable learning |
| Total Variation (TV) |  | Penalizes differences between neighboring parameters | Smooths solutions by reducting oscillations | Used in image processing tasks |
| Frobenius norm |  | Penalizes large values across all elements of a matrix (e.g. weight matrices in neural networks) | It encourages smaller and smoother weights in the matrix, preventing overfitting by controlling the overall scale of weights | Particularly relevant in models involving matrix operations, such as regularizing weight matrices in NNs, collaborative filtering (e.g. matrix factorization models), multi-task learning, wehere each tasks parameters are organized in matrices |