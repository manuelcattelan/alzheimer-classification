from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import time


dt_parameters = {"criterion": ["gini", "entropy"],
                 "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 "min_samples_split": list(np.arange(2, 20, 1)),
                 "min_samples_leaf": list(np.arange(1, 20, 1)),
                 "min_impurity_decrease": list(np.arange(0.0, 0.5, 0.1))}


rf_parameters = {}


svc_parameters = {}


def tune_classifier(classifier,
                    cross_validator,
                    df,
                    tune_mode,
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
                                       cv=cross_validator)
        case "grid":
            tuner = GridSearchCV(estimator=classifier,
                                 param_grid=tune_parameters,
                                 cv=cross_validator)
    start = time.time()
    # Tune parameters on dataframe
    tuner.fit(X, y)

    return (tuner.best_estimator_,
            tuner.best_estimator_.get_params(),
            time.time() - start)