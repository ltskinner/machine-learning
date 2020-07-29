# Be Careful: Time Series is Tricky

## Accuracy Metrics

### R2 Score

#### Can be VERY misleading

* aka the **coefficient of determination**
* Theres a couple different ways to calc it
  * Not all are equivalent haha

### Mean Absolute Pecentage Error

#### Can be VERY misleading as well

## Auto-correlation

* Is the correlation of a signal with a delayed copy of itself
* Index "t+1" is likely to be near "t"
  * Basically uses the previous value as the prediction for the next

## Random Walks

### Are you walking randomly?

* Time series shows a strong temporal dependence that decays linearly or in a similar pattern
  * **Autocorrelation**
* The time series is non-stationary and making it stationary shows no obviously learnable structure in the data
* The **persistence model** provides the best source of reliable predictions
  * Next step is the last step

### Use baseline forcasts with persistence models to test if a random walk
