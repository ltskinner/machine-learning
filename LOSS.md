# Loss

- Normalized Temperature-scaled Cross-Entropy Loss
  - Discovered in context of SimCSE (Sentence Contrastive Learning)

## Regularization

A technique to prevent overfitting by adding a penalty term (like L1 or L2) to the loss function, discouraging overly complex models.

controls model complexity

| Penalty Term | mathematical Form | Key Concept | Effect on Model | Use Case |
| - | - | - | - | - |
| L1 (Lasso) |  | Encourages sparsity by penalizing the sum of absolute values of weights | Drives some weights to exactly zero, performing feature selection | Useful when only a few features are important |
| L2 (Ridge) |  | Penalizes large weights by summing their squares | Shrinks weights smoothly but keeps them non-zero | Suitable when all features have some relevance, helping with multicollinearity |
| Elastic Net |  | Combines L1 and L2 regularization | Balances sparsity (L1) and shrinkage (L2) | Ideal for models where some features are redundant and others sparse |
| Group Lasso |  | Applies L1 regularization on groups of related features | Selects or discards entire feature groups | Useful when features are grouped and dependent |
| Dropout |  | Randomly sets a fraction of weights to zero during training | Introduces randomness to reduct overfitting and prevents neurons from co-adapting | Wiedely used in deep learning networks |
| Max-Norm |  | Constrains the magnitude of weight norms | Prevents exploding weights | Common in neural networks for stable learning |
| Total Variation (TV) |  | Penalizes differences between neighboring parameters | Smooths solutions by reducting oscillations | Used in image processing tasks |
