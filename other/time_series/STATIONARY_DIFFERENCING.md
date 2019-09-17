# Stationary Time Series and Differencing

Predicting the *difference* in values in between time steps instead of the value itself is better

## Stationary Definition

A stationary time series is one whos properties do not depend on the time at which the series is observed

* Stationary conditions that stay constant over time
  * Mean
  * Variance
  * Autocorrelation
* Non-Stationary conditions
  * Trending data
  * Seasonal data
  * Cyclic data that is dependent solely on time
    * Not envionrmental conditions

### [Visualization](https://otexts.com/fpp2/stationarity.html)

## Differencing Definition

Calculate the differences between consecutive observations

* Used to make non-stationary time series stationary
* Eliminates or reduces:
  * Trend
  * Seasonality

## Detection

### ACF Plots

[Roll-your-own Example](https://towardsdatascience.com/significance-of-acf-and-pacf-plots-in-time-series-analysis-2fa11a5d10a8)
