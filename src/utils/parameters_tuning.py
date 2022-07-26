from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import time


dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "min_samples_split": list(np.arange(2, 20, 1)),
        "min_samples_leaf": list(np.arange(1, 20, 1)),
        "min_impurity_decrease": list(np.arange(0.0, 0.5, 0.1))
        }


def tune_classifier(clf, cv, df, parameters, mode):
    # create two subframes containing only model features and model label
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # based on mode argument, run randomized search or grid search
    match mode:
        case "randomized":
            tuner = RandomizedSearchCV(
                    estimator=clf,
                    param_distributions=parameters,
                    cv=cv,
                    )
        case "grid":
            tuner = GridSearchCV(
                    estimator=clf,
                    param_grid=parameters,
                    cv=cv,
                    )
    # time parameter tuning
    start = time.time()
    # start parameter tuning
    tuner.fit(X, y)

    return tuner.best_estimator_, tuner.best_params_, time.time() - start
