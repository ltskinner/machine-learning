# Scaling

## Conform Values to a Range

### Scale to a Range

```python
from sklearn.preprocessing import MinMaxScaler

# Scale between 0 and 1
mm_scaler = MinMaxScaler(feature_range(0, 1))
mm_scaler.fit(array_of_ints)
mm_scaler.fit_transform(
    np.array(list_of_ints).reshape(-1, 1))
```

### Scale to Guassian

* `mean` of 0
* `std` of 1

```python
gauss = preprocessing.scale(list_of_ints)

gauss.mean()
gauss.std()
```

## Binarize Data

```python
from sklearn.preprocessing import Binarizer

# Want data above -25 to be activated
THRESH = -25
tz = Binarizer(threshold=THRESH).fit([array_of_ints])
tz.transform([array_of_ints])
```
