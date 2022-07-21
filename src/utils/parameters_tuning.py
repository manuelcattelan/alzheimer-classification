from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

param_distribution = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1, 10, 1)),
        'min_samples_split': list(range(2, 20, 1)),
        'min_samples_leaf': list(range(1, 20, 1)),
        'min_impurity_decrease': list(np.arange(0.0, 1.0, 0.1))
        }


param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 3, 5, 10],
        'min_samples_split': [2, 3, 5, 10, 15, 20],
        'min_samples_leaf': [1, 3, 5, 10, 15, 20],
        'min_impurity_decrease': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }


def tune_classifier(clf, cv, parameters, df, mode):
    adjacent_feature_columns = ["DurationTot", "Instruction"]
    features = np.r_[
            df.columns.get_loc(adjacent_feature_columns[0]):
            df.columns.get_loc(adjacent_feature_columns[1]) + 1
            ]
    # get model feature names and label name
    model_features = df.columns[features]
    model_label = df.columns[-1]

    # create two subframes containing only model features and model label
    X = df[model_features]
    y = df[model_label]

    match mode:
        case 'randomized':
            s = RandomizedSearchCV(
                    estimator=clf,
                    param_distributions=parameters,
                    cv=cv,            
                    )
        case 'grid':
            s = GridSearchCV(
                    estimator=clf,
                    param_grid=parameters,
                    cv=cv,
                    )

    start = time.time()
    s.fit(X, y)

    return s.best_estimator_, time.time() - start
