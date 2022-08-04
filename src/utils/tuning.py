from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform


DT_PARAM_DISTRIBUTION = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": randint(low=2, high=40),
        "min_samples_leaf": randint(low=1, high=20),
        "min_impurity_decrease": uniform(0.1, 0.5)
        }


DT_PARAM_GRID = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]
        }


RF_PARAM_DISTRIBUTION = {
        "n_estimators": randint(10, 150),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": randint(low=2, high=40),
        "min_samples_leaf": randint(low=1, high=20),
        "min_impurity_decrease": uniform(0.1, 0.5),
        "bootstrap": [True, False]
        }


RF_PARAM_GRID = {
        "n_estimators": [10, 30, 50, 100, 150],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5],
        "bootstrap": [True, False]
        }


SVC_PARAM_DISTRIBUTION = {
        "C": loguniform(1e-1, 1e3),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": randint(1, 5),
        "gamma": loguniform(1e-3, 1e2),
        "probability": [True, False]
        }


SVC_PARAM_GRID = {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4, 5],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "probability": [True, False]
        }


def tune_clf_params(
        clf,
        df,
        tune_mode,
        tune_parameters,
        tune_iterations,
        tune_metric,
        n_jobs
        ):
    # Divide dataframe into two subframes:
    # X contains all feature columns except for the Id and Label column
    # y contains the Label column only
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    if tune_mode == "randomized":
        # Initialize RandomizedSearch tuner
        tuner = RandomizedSearchCV(
                estimator=clf,
                param_distributions=tune_parameters,
                n_iter=tune_iterations,
                scoring=tune_metric,
                n_jobs=n_jobs,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=0
                    ),
                random_state=0
                )
    elif tune_mode == "grid":
        # Initialize GridSearchCV tuner
        tuner = GridSearchCV(
                estimator=clf,
                param_grid=tune_parameters,
                scoring=tune_metric,
                n_jobs=n_jobs,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=0
                    )
                )

    # Tune provided parameters on dataframe
    tuner.fit(X, y)

    return tuner.best_estimator_, tuner.cv_results_
