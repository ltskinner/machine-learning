# Chapter 4: Interpretable Models

"The easiest way to achieve interpretability is to use only a subset of algorithms that create interpreteable models"

| Algorithm | Linear | Monotone | Interaction | Task |
| - | - | - | - | - |
| Linear Regression | Yes | Yes | No | regr |
| Logistic regression | No | Yes | No | class |
| Decision trees | No | Some | Yes | class, regr |
| RuleFit | Yes | No | Yes | class, regr |
| Naive Bayes | No | Yes | No | class |
| k-nearest neighbors | No | No | No | class, regr |

## 4.1 Linear Regression

* Linear regression models predict the target as a weighted sum of the feature inputs
  * These have highly linear relationships

### Traits

* Linearity
  * Obviously, these models are a linear combination of features
  * This is its greatest strength, and greatest weakness
* Normality
  * It is assumed that the target outcome given the feature follows a normal distribution - if this is violated, feature weights are invalid
* Homoscedasticity (constant variance)
  * The variance of the error terms is assumed to be constant
* Independence
  * It is assumed that each instance is independent of any other instance
  * An example of NOT independent, is multiple blood tests per patient - the data points are not independent
* Fixed features
  * Input features are considered "fixed" aka they are treated as given constants
  * As opposed to questionable measurements
* Abscence of multicollinearity
  * Dont want strongly correlated features because it messes up estimation of the weights

### 4.1.1 Interpretation

* Numerical feature:
  * Increaseing the numerical feature by one unit changes the estimated outcome by its weight
* Binary feature:
  * A feature takes one of two possible values for each instance
  * Changing between categories adjusts the estimated outcome by the feature weight
* Categorical feature with multiple categories
  * Typically, one hot encode to a bunch of binary features lmao
* Intercept 🅱️0
  * The base value that all other features add or remove from
  * If all feature weights are zero, the output would just be the weight of this Intercept

#### Actual interpretation (lmao)

* Interpretation of a Numerical Feature
  * Increase of feature xk by one unit increases the prediction for y by 🅱️k units when all other feature values remain fixed
* Interpretation of a Categorical Feature
  * " ^^ "
  * R-squared is also important - want a high R squared (model explains the data nicely)
  * Also, use the `adjusted R-squared` because it accounts for the number of features used in the model
* Feature Importance
  * Easily measured by the absolute value of its t-statistic

--------------- stopped at examples -----------------
