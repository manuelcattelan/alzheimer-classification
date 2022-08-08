from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


DT_PARAM_GRID = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5]
        }


SVC_PARAM_GRID = {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4, 5],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "probability": [True, False]
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


def tune_clf_params(
        clf,
        df,
        tune_parameters,
        n_jobs
        ):
    # Divide dataframe into two subframes:
    # X contains all feature columns except for the Id and Label column
    # y contains the Label column only
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Initialize GridSearchCV tuner
    tuner = GridSearchCV(
            estimator=clf,
            param_grid=tune_parameters,
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
