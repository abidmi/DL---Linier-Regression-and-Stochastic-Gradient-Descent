# This repository about Linear Regression and Stochastic Gradient Descent

## Motivation of Experiment
- Further understand of linear regression ，closed-form solution and Stochastic gradient descent.
- Conduct some experiments under small scale dataset.
- Realize the process of optimization and adjusting parameters.
## Dataset
Linear Regression uses [Housing](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing) in [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), including 506 samples and each sample has 13 features. You are expected to download scaled edition. After downloading, you are supposed to divide it into training set, validation set.
## Environment for Experiment
[Python3](https://www.python.org/), at least including following python package: [sklearn](http://scikit-learn.org/stable/)，[numpy](http://www.numpy.org/)，[jupyter](http://jupyter.org/)，[matplotlib](https://matplotlib.org/)
It is recommended to install [anaconda3](https://anaconda.org/) directly, which has built-in python package above.
## Experiment Step

*closed-form solution of Linear Regression*

- Load the experiment data. You can use [load_svmlight_file](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html) function in sklearn library.
- Devide dataset. You should divide dataset into training set and validation set using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function. Test set is not required in this experiment.
- Select a Loss function.
- Get the formula of the closed-form solution, the process details the courseware ppt.
- Get the value of parameter ***W*** by the closed-form solution, and update the parameter ***W***.
- Get the ***Loss***, ***loss_train*** under the training set and ***loss_val*** by validating under validation set.
- Output the value of ***Loss***, ***loss_train*** and ***loss_val***.

*Linear Regression and Stochastic Gradient Descent*

- Load the experiment data. You can use [load_svmlight_file](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html) function in sklearn library.
- Devide dataset. You should divide dataset into training set and validation set using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function. Test set is not required in this experiment.
- Initialize linear model parameters. You can choose to set all parameter into zero, initialize it randomly or with normal distribution.
- Choose loss function and derivation: Find more detail in PPT.
- Calculate ***G*** toward loss function from each sample.
- Denote the opposite direction of gradient ***G*** as ***D***.
- Update model: ***Wt=Wt−1+ηD. η***. is learning rate, a hyper-parameter that we can adjust.
- Get the ***loss_train*** under the training set and ***loss_val*** by validating under validation set.
- Repeate step 5 to 8 for several times, andand output the value of ***loss_train*** as well as ***loss_val***.