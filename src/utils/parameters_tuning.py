from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import time


dt_parameters = {"criterion": ["gini", "entropy"],
                 "splitter": ["best", "random"],
                 "max_depth": [None, 1, 3, 5, 8, 10],
                 "min_samples_split": [2, 5, 10, 15, 20],
                 "min_samples_leaf": [1, 5, 10, 15, 20],
                 "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]}


rf_parameters = {"n_estimators": [50, 100, 150, 200, 250],
                 "criterion": ["gini", "entropy"],
                 "max_depth": [None, 1, 3, 5, 8, 10],
                 "min_samples_split": [2, 5, 10, 15, 20],
                 "min_samples_leaf": [1, 5, 10, 15 ,20],
                 "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5],
                 "bootstrap": [True, False]}


svc_parameters = {"C": [0.1, 1, 10, 100, 1000],
                  "kernel": ["linear", "poly", "rbf"],
                  "degree": [1, 2, 3, 4, 5],
                  "gamma": ["scale", "auto", 0.0001, 0.001, 0.1, 1, 10, 100]}


def tune_classifier(classifier,
                    cross_validator,
                    df,
                    jobs,
                    tune_mode,
                    tune_iter,
                    tune_parameters):
    # Separate dataframe into two subframes:
    # X contains all feature columns except for the label column
    # y contains only the label column
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Based on tune_mode, initialize tuner
    # to RandomizedSearch or GridSearch
    match tune_mode:
        case "randomized":
            tuner = RandomizedSearchCV(estimator=classifier,
                                       param_distributions=tune_parameters,
                                       cv=cross_validator,
                                       n_iter=tune_iter,
                                       n_jobs=jobs)
        case "grid":
            tuner = GridSearchCV(estimator=classifier,
                                 param_grid=tune_parameters,
                                 cv=cross_validator,
                                 n_jobs=jobs)
    start = time.time()
    # Tune parameters on dataframe
    tuner.fit(X, y)

    return (tuner.best_estimator_,
            tuner.best_params_,
            time.time() - start)
