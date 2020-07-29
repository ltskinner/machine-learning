# Input Formatting

## skelarn

### Encoding types

* Categorical encoding
  * NOT one-hot
  * Nominal values whose categories have no relation
* Ordinal encoding
  * Variables have a clear ordering
  * Low, medium, high
* Interval
  * lower_bound < value < upper_bound

### Categorical Encoding

```python
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Encode strings to unique values between 0-n
l_enocder = preprocessing.LabelEncoder()
l_encoder.fit(['foo', 'bar', 'z'])

# Retrieve values
foo_val = l_encoder.transform(['foo'])

# Create dictionary of mappings
dict(
  zip(l_encoder.classes_, l_encoder.transform(l_encoder.classes_))
)
```

#### Encoding a df

```python
l_encoder = preprocessing.LabelEncoder()
df = pd.dataFrame(string_data_dict)

encoded_df = df.apply(l_encoder.fit_transform)
```

### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

o_encoder = OrdinalEncoder()
o_encoder.fit([
    ['foo', 'bar', 'baz'],
    ['x', 'y', 'z']
  ])
o_encoder.transform([
  ['foo', 'y']
])
# --> [[0, 1]]
```

### One-Hot Encoding

* Good for regressors and SVMs

```python
from sklearn.preprocessing import OneHotEncoder

h_encoder = OneHotEncoder(handle_unknown='ignore')

h_encoder.fit(array_2d)
h_encoder.transform(target_data).toarray()
```

#### One-Hot Encoding a df

```python
# Creates new columns for each distinct value in a column
pd.get_dummies(df)
```
