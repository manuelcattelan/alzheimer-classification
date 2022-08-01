from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform
import pandas as pd


DT_PARAM_DISTRIBUTION = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 1, 2, 3, 4, 5],
        "min_samples_split": randint(low=2, high=40),
        "min_samples_leaf": randint(low=1, high=20),
        "min_impurity_decrease": uniform(0.1, 0.5)
        }


RF_PARAM_DISTRIBUTION = {
        "n_estimators": randint(10, 150),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 1, 2, 3, 4, 5],
        "min_samples_split": randint(low=2, high=40),
        "min_samples_leaf": randint(low=1, high=20),
        "min_impurity_decrease": uniform(0.1, 0.5),
        "bootstrap": [True, False]
        }


SVC_PARAM_DISTRIBUTION = {
        "C": loguniform(1e-1, 1e3),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": randint(1, 10),
        "gamma": loguniform(1e-4, 1e2),
        "probability": [True, False]
        }


def tune_clf_params(
        clf,
        df,
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

    # Initialize RandomizedSearch tuner
    tuner = RandomizedSearchCV(
            estimator=clf,
            param_distributions=tune_parameters,
            n_iter=tune_iterations,
            scoring=tune_metric,
            n_jobs=n_jobs,
            cv=StratifiedKFold(
                n_splits=10,
                shuffle=True,
                random_state=0
                ),
            random_state=0
            )

    # Tune provided parameters on dataframe
    tuner.fit(X, y)
    # Build dataframe containing tuning results
    tuner_results = pd.DataFrame(tuner.cv_results_)[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "mean_score_time",
                "rank_test_score",
                ]
            ]
    # Sort dataframe by best ranking parameters
    tuner_results = tuner_results.sort_values("rank_test_score")

    return tuner.best_estimator_, tuner_results
