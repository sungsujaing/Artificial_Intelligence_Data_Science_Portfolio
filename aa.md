In many case, different algorithms may exhibit similar performances. What is more important includes nature of data, amount of data, hyperparameter tunning, etc.

## Linear models
* Common parametric algorithm for both regression and classification tasks.
* In case of linear regression, a simple normal equation can be solved instead of applying the gradient descent optimizer. However, this numerical approach tends to be slower when n gets large.
* Usually sigmoid function is applied to a linear regression model to build a logistic regression classifier.
* Important hyperparameters include
  * Regularization parameter
  * learning rate

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
