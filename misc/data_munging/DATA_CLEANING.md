# Data Cleaning

## Understanding the Dataset

### Index

```python
# Check the index values
df.index.values

# Check if a certain index exists
'foo' in df.index.values

# Check if unique items
df['id'].is_unique

# If index does not exist
df.set_index('column_to_use', inplace=True)
```

#### Accessing data with .loc

```python
# After setting an index, can use loc to grab rows
df.loc[unique_identifier]
```

#### Accessing data with .iloc

```python
# Can always get by position with .iloc
df.iloc[0]  # Grabs the first value
```

### Columns

```python
# Print data types
df.dtypes

# Ensure columns are unique
for col in df.columns:
    print(col, df[col].is_unique)

# Drop Columns
df.drop(list_of_cols, inplace=True, axis=1)
# or
df.drop(columns=list_of_cols, inplace=True)
```

#### Renaming Columns

```python
column_name_map = {
    'Original_1': 'New_1',
    'Original_2': 'New_2
}

df = df.rename(columns=column_name_map, inplace=True)
```

### NaNs

* Can replace with
  * mean
  * median
* Or just remove
  * it depends...

#### Fill With Deliberate Values

```python
value = 0
df['col'] = df['col'].fillna(value)

mean_value = df['col'].mean()
df['col'] = df['col'].fillna(mean_value)
```

#### Fill by Propagation

```python
# Fill next 1 value
df.fillna(method='pad', limit=1)

# Fill backwards
df.fillna(method='bfill)
```

### Dropping

```python
# Drop ROWS with NaNs
df.dropna()

# Drop COLUMNS with NaNs
df.dropna(axis=1)

# Drop COLUMNS whose rows arent above a threshold
THRESH_90 = int(df.shape[0] * 0.9)
df.dropna(thresh=THRESH_90, axis=1)
```

## Feature Engineering

### Adding Columns by Conditionals

```python
# Syntax
col = np.where(if_this_condition, do_this, else_this)

# Example
df['can_drink'] = np.where(df['age'] > 21, True, False)

# Stacking
df['number_name'] = np.where(df['value'] == 0, 'Zero',
                        np.where(df['value'] == 1, 'One',
                            np.where(df['value'] == 2, 'Two', 'Three')))
```

### applymap

```python
def select_before_char(item):
  if '[' in item:
    return item[:item.find('[')]

df = df.applymap(select_before_char)
```

## Asserting and Testing Data Expectations

```python
# Assert all values in a column are >= 0
assert(df['col'] >= 0).all()  # Should return nothing

# Assert that none of the values are str
assert(df['col'] != str).any()
```

* `.all()`
  * Ensures that every single element passes the assert test
* `.any()`
  * Sees if any of the elements pass the assert test

### Other things to check

* For **negative** values
* Ensure two columns are the same
* Determine the results of a transformation
* Ensure IDs are correct and unique

#### [More Assert Info](https://www.mattcrampton.com/blog/a_list_of_all_python_assert_methods/)

## String Parsing

### Emails and Urls

```python
from beautifier import Email, Url

# Email
email_string = 'foo@bar.com'
email = Email(email_string)

print(email.domain)
print(email.username)
print(email.is_free_email)

# Url
url_string = 'https://github.com/labtocat/beautifier'
url = Url(url_string)

print(url.param)
print(url.username)
print(url.domain)
```

### Unicode

#### MojiBake Handling

```python
import ftfy

mojibake = '&macr;\\_(ã\x83\x84)_/&macr; \ufeffParty \001\033[36;44mI&#x92;m'

print(ftfy.fix_text(mojibake))
```

### Deduplication and Entity Recognition

* [dedpue](https://medium.com/district-data-labs/basics-of-entity-resolution-with-python-and-dedupe-bc87440b64d4)
  * Actually go and leverage this
* Works better with preprocessed text
  * search ['preProcess'](https://medium.com/@rrfd/cleaning-and-prepping-data-with-python-for-data-science-best-practices-and-helpful-packages-af1edfbe2a3)

### Fuzzy String Matching

* Main uses are:
  * Ratios
  * Extracting words given a seed
* [Examples](https://github.com/seatgeek/fuzzywuzzy)
