>In many case, different algorithms may exhibit similar performances. What is more important includes nature of data, amount of data, hyperparameter tunning, etc.

#### For the list below:
**_m_** refers to # of training data<br>
**_n_** refers to # of feature

## Linear models
* Common parametric algorithm for both regression and classification tasks.
* In case of linear regression, a simple normal equation can be solved instead of applying the gradient descent optimizer. However, this numerical approach tends to be slower when n gets large.
* Usually **sigmoid function** is applied to a linear regression model to build a logistic regression classifier.
* No hyperparameter that controls model complexity 
* Important hyperparameters include
  * Regularization parameter
  * learning rate
  
$\sqrt{3x-1}+(1+x)^2$

| Pros | Cons |
| ------ | ------ |
| <ul><li>Fast training and simple to understand/explain</li><li>Soft classifier in logistic regression</li><li>Provide feature significance</li><li>Convex cost function</li><li>Work well when m is small and n is large</li></ul> | <ul><li>Not flexible to deal with non-linear hyperplanes</li><li>Adding high-order features is difficult and time-consuimng</li><li>Susceptible to outliers and co-linearity</li></ul>|

## Support Vector Machines (SVM)
* Common parametric algorithm for both regression and classification tasks.
* For classification task, it is a hard classifier with a maximum margin hyperplane.
* Its cost function is the simpler version of that of logistic regression.

| Pros | Cons |
| ------ | ------ |
| <ul><li>Support non-linear problems well using kernels</li><li>Robust to overfitting and outliers</li><li>Convex cost function</li><li>Work well when m is small and n is large</li></ul> | <ul><li>Memory intensive</li><li>Possibly be difficult to tune hyperparameters</li></ul>|

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
