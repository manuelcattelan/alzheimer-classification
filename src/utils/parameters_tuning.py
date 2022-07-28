from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
import numpy as np
import time


dt_parameters = {"criterion": ["gini", "entropy"],
                 "splitter": ["best", "random"],
                 "max_depth": [None, 1, 2, 3, 4, 5],
                 "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                 "min_samples_leaf": [1, 3, 5, 8, 10, 12, 14, 16, 18, 20],
                 "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]}


dt_distribution = {"criterion": ["gini", "entropy"],
                   "splitter": ["best", "random"],
                   "max_depth": [None, 1, 2, 3, 4, 5],
                   "min_samples_split": randint(low=2, high=20),
                   "min_samples_leaf": randint(low=1, high=20),
                   "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]}


rf_parameters = {"n_estimators": [10, 100, 200, 300, 400, 500],
                 "criterion": ["gini", "entropy"],
                 "max_depth": [None, 1, 2, 3, 4, 5],
                 "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                 "min_samples_leaf": [1, 3, 5, 8, 10, 12, 14, 16, 18, 20],
                 "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5],
                 "bootstrap": [True, False]}


rf_distribution = {"n_estimators": randint(10, 250),
                   "criterion": ["gini", "entropy"],
                   "max_depth": [None, 1, 2, 3, 4, 5],
                   "min_samples_split": randint(2, 20),
                   "min_samples_leaf": randint(2, 20),
                   "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5],
                   "bootstrap": [True, False]}


svc_parameters = {"C": [1, 10, 100, 1000],
                  "kernel": ["linear", "poly", "rbf"],
                  "degree": [1, 2, 3, 4, 5],
                  "gamma": ["scale", "auto", 0.0001, 0.001, 0.1, 1, 10, 100]}


svc_distribution = {"C": loguniform(1e0, 1e3),
                    "kernel": ["linear", "poly", "rbf"],
                    "degree": randint(1, 5),
                    "gamma": loguniform(1e-4, 1e2)}


def tune_classifier(clf,
                    df,
                    tune_mode,
                    tune_iter,
                    tune_parameters,
                    tune_metric,
                    n_jobs):
    # Separate dataframe into two subframes:
    # X contains all feature columns except for the label column
    # y contains only the label column
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Based on tune_mode, initialize tuner
    # to RandomizedSearch or GridSearch
    match tune_mode:
        case "randomized":
            tuner = RandomizedSearchCV(estimator=clf,
                                       param_distributions=tune_parameters,
                                       n_iter=tune_iter,
                                       scoring=tune_metric,
                                       n_jobs=n_jobs,
                                       cv=StratifiedKFold(shuffle=True,
                                                          random_state=0),
                                       random_state=0)
        case "grid":
            tuner = GridSearchCV(estimator=clf,
                                 param_grid=tune_parameters,
                                 scoring=tune_metric,
                                 n_jobs=n_jobs,
                                 cv=StratifiedKFold(shuffle=True,
                                                    random_state=0))
    # Tune parameters on dataframe
    start = time.time()
    tuner.fit(X, y)
    stop = time.time()

    tune_time = stop - start

    return (tuner.best_estimator_,
            tuner.best_params_,
            tune_time)
