# Comparative studies of<br>Classic supervised machine learning algorithms

> Learn from experience (E) with respect to some task (T) and some performance measure (P), if its performance (T), as measured by (P), improves with expereince (E) - Tom Mitchell (1998)

In many case, different Machine Learning (ML) algorithms may exhibit similar performances. What is more important includes nature of data, amount of data, hyperparameter tunning, etc.

#### For the list below:
**_m_** refers to # of training data<br>
**_n_** refers to # of feature

## Linear models
#### Linear regression
* Common parametric algorithm for regression tasks
* Model: h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>x where &theta; and x are n+1 dimensional vectors
* Typical cost function: MSE
* In case of linear regression, a simple normal equation (&theta; = (X<sub>T</sub>X)<sup>-1</sup>X<sup>T</sup>y) can be solved instead of applying the gradient descent optimizer. However, this numerical approach tends to be slower when **_n_** gets large.
#### Logistic regression
* Common parametric algorithm for classification tasks
* Usually **sigmoid function** (g) is applied to a linear regression model to build a logistic regression classifier
* Model: h<sub>&theta;</sub>(x) = g(&theta;<sup>T</sup>x) where g(z) = (1+e<sup>-z</sup>)<sup>-1</sup>
* Typical cost function: entropy
* Falling in 0 - 1, logistic regression predict probability y falling in a certain class given x, parameterized by &theta;
* Threshold (default: 0.5) can be arbitrarily adjusted based on the specific evaluation criteria
#### For both cases
* No hyperparameter that controls model complexity 
* Important hyperparameters include
  * Regularization parameter (&lambda;) - should followed by feature scaling
  * Learning rate (&alpha;)

| Pros | Cons |
| ------ | ------ |
| <ul><li>Fast training and easy to understand/explain the results</li><li>Provide feature significance</li><li>**Convex cost function**</li><li>Work well when **_m_** is small and **_n_** is large</li></ul> | <ul><li>Naturally **_not_** flexible to model non-linear hyperplanes</li><li>Cannot handle categotical variables well</li><li>Adding high-order features is an option, but is difficult and time-consuimng</li><li>Feature scaling becomes important when adding high-order features</li><li>Susceptible to outliers and co-linearity</li></ul> |

## Support Vector Machines (SVM)
* Common parametric algorithm for both regression and classification tasks
* Only uses the subset of training data (support vectors) to model decision functions
* Predict 1 if &theta;<sup>T</sup>x >= 1. Predict 0 if &theta;<sup>T</sup>x <= -1, forcing to have an extra margin
* Its cost function is the simpler version of that of logistic regression
* For classification task, it is a hard classifier which provides no probability
* Using kernel tricks, feature space is expanded to higher orders without adding any new features. A linear classifier fitted in the transformed space becomes non-linear classifier in the original space.
* Important hyperparameters include
  * Regularization parameter (C)
  * Type of kernel
  * Kernel-related parameters such as kernel width (&Gamma;) in case of using RBF kernel, and degree of polynomial in case of using polynomial kernel.

| Pros | Cons |
| ------ | ------ |
| <ul><li>Support non-linear problems well using kernels</li><li>Robust to overfitting and outliers compared to the linear models</li><li>Convex cost function</li><li>Work well when **_m_** is small and **_n_** is large (even infinite!)(</li><li>Memory efficient</li></ul> | <ul><li>Difficult to interpret the restuls</li><li>When using kernels, possibly be difficult to tune hyperparameters</li><li>With **_m_** going large, training speed and memory efficiency goes down greatly</li><li>Feature scaling becomes criticial when employing kernels</li></ul> |

## K-nearest Neighbors (KNN)
* Non-parametric algorithm for both regression and classification tasks.
* For classification task, it is a hard classifier with a maximum margin hyperplane.
* Its cost function is the simpler version of that of logistic regression.

| Pros | Cons |
| ------ | ------ |
| <ul><li>Very simple</li><li>Involve only a few hyperparameters</li><li>Support non-linear problems</li><li>May outperform other complex algorithm when m is large and n is small</li></ul> | <ul><li>'K' should be wisely selected</li><li>High prediction cost when m is large</li><li>Hard to deal with categorical features</li><li>Feature scailing is important</li></ul>|

## Decision Tree
* Non-parametric algorithm for both regression and classification tasks.

| Pros | Cons |
| ------ | ------ |
| <ul><li>Easily learn non-linear solution</li><li>Fairly robust to outliers and co-linearity problems</li><li>No feature pre-processing is required</li><li>Easy to explain the rationale</li><li>Deals with categorical data very well</li></ul> | <ul><li>May lose important information while dealing with continuous variable</li><li>Overfit quite easily</li></ul>|

## Ensemble Tree (Bagging and Boosting)
* Non-parametric algorithm for both regression and classification tasks.
* More robust and accurate compared to the single tree model.

| Pros | Cons |
| ------ | ------ |
| <ul><li>Easily learn non-linear solution</li><li>Fairly robust to outliers and co-linearity problems</li><li>No feature pre-processing is required</li><li>Handles overfitting issue very efficiently</li><li>Deals with categorical data very well</li></ul> | <ul><li>May lose important information while dealing with continuous variable</li><li>Computationally expensive as the number of trees gets larger</li><li>Difficult to explain the results</li></ul>|

## Naive Bayes
* Generative algorithm for classification tasks
* Simple algorithm that depends on Bayes rule

| Pros | Cons |
| ------ | ------ |
| <ul><li>Performs very well compared to its simplicity</li><li>Very simple and fast to train</li><li>Works well when m is small</li></ul> | <ul><li>Assume mutual independence among features</li><li>Highly susceptible to co-linearity</li><li>Easily outperformed by other properly-tuned complex models</li></ul>|

## Neural network
* Generative algorithm for classification tasks
* Simple algorithm that depends on Bayes rule

| Pros | Cons |
| ------ | ------ |
| <ul><li>Can learn very complex functions</li><li>No need for arbitrary feature engineering</li></ul> | <ul><li>Require large amount of data to be trained properly</li><li>Computationally expensive</li><li>Many hyperparameteres to tune</li></ul>|
<li></li>
