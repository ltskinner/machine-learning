# Scaling

[General Preprocessing from SciKit](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)

## Conform Values to a Range

| Preprocess | Function | When to Use |Range | Mean | Distribution |
| ------------- | ------------- | -----| ---- | ---- | --- |
| Scale | MinMaxScaler | Use first | 0-1 (can override) | Varies | Bounded |
| Standardize | RobustScaler | Reduce influence of **outliers** | Varies | Varies | Unbounded |
| Standardize | StandardScaler | Convert to **normally distributed** | Varies | 0 | Unbounded, Unit variance |
| Normalize | Normalizer | On **rows** | Varies | 0 | Unit norm |

### MinMaxScaler

* *Subtracts* the **minimum value**, then *divides* by the **range**
* Preserves the shape of the original distribution
  * Doesnt reduce the importance of outliers

```python
from sklearn import preprocessing

# Scale between 0 and 1
mm_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
mm_scaler.fit_transform(
    np.array(list_of_ints).reshape(-1, 1))
```

### RobustScaler

* *Subtracts* the **median** and *divides* by the **interquartile range**
  * 75% - 25% values
* Use to reduce the effect of outliers
* Does not scale to predetermined range

```python
r_scaler = preprocessing.RobustScaler()
r_scaler.fit([array_of_ints])
r_Scaler.transform([array_of_ints])
```

### StandardScaler

* *Subtracts* the **mean**, then scales to **unit variance**
  * **unit variance** *divides* all the values by the standard deviation
* Makes the mean of the distribtuion 0
  * 68% of values will lie between -1 and 1
* Scales to Guassian
  * `mean` of 0
  * `std` of 1

```python
s_scaler = preprocessing.StandardScaler()
s_scaler.fit([list_of_ints])
s_scaler.transform([list_of_ints])

s_scaler.mean_
s_scaler.std_
```

#### Function based option

```python
gauss = preprocessing.scale(list_of_ints)
gauss.mean()
gauss.std()
```

### Normalizer

* Normalizer works on the **rows** NOT the columns

## Binarize Data

```python
# Want data above -25 to be activated
THRESH = -25
tz = preprocessing.Binarizer(threshold=THRESH).fit([array_of_ints])
tz.transform([array_of_ints])
```
