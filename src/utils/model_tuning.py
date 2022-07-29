from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
import numpy as np
import time


dt_distribution = {"criterion": ["gini", "entropy"],
                   "max_depth": randint(1, 10),
                   "min_samples_split": randint(low=2, high=50),
                   "min_samples_leaf": randint(low=1, high=50),
                   "max_features": randint(low=1, high=100),
                   "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]}


rf_distribution = {"n_estimators": randint(10, 500),
                   "criterion": ["gini", "entropy"],
                   "max_depth": randint(1, 10),
                   "min_samples_split": randint(2, 50),
                   "min_samples_leaf": randint(2, 50),
                   "max_features": randint(low=1, high=100),
                   "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5],
                   "bootstrap": [True, False]}


svc_distribution = {"C": loguniform(1e0, 1e3),
                    "kernel": ["linear", "poly", "rbf"],
                    "degree": randint(1, 5),
                    "gamma": loguniform(1e-4, 1e2),
                    "probability": [True, False]}


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

    # Initialize RandomizedSearch tuner
    tuner = RandomizedSearchCV(estimator=clf,
                               param_distributions=tune_parameters,
                               n_iter=tune_iter,
                               scoring=tune_metric,
                               n_jobs=n_jobs,
                               cv=StratifiedKFold(shuffle=True,
                                                  random_state=0),
                               random_state=0)
    # Tune parameters on dataframe
    start = time.time()
    tuner.fit(X, y)
    stop = time.time()

    tune_time = stop - start

    return (tuner.best_estimator_,
            tuner.best_params_,
            tune_time)
