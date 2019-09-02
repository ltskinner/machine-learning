# Principal Component Analysis

## Process

* Be sure to Scale values
* Can use several `svd_solver` parameters
  * Stick with `svd_solver="auto"` for the most part tho

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline

std_pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, svd_solver='auto'),
    GaussianNB())

std_pca_pipeline.fit(X, y)

# Extract PCA from pipeline
pcs_std = std_pca_pipeline.named_steps['pca']

# Components
pca_std.components_[0]

# Transform
# Note, X wont have the StandardScaler()
X_pca = pca_std.transform(X)
```

### Variance Ratio

```python
pca_std.explained_variance_ratio_
# --> [0.72... 0.23...]
# 0.72 + 0.23 = 95% of information contained in 2 components

# To select the min number of features
pca = PCA(0.95)
```

## Visualize

* [Example](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
