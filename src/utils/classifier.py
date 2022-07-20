from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import time


def train_classifier(clf, X, y, train_index):
    # define training subframe for current split training index
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # train classifier and time training
    start = time.time()
    trained_clf = clf.fit(X_train, y_train)
    stop = time.time()

    train_time = stop - start

    return trained_clf, train_time


def test_classifier(clf, X, y, test_index):
    # define testing subframe for current split testing index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # make predictions on test data and time testing
    start = time.time()
    y_pred = clf.predict(X_test)
    stop = time.time()

    test_time = stop - start

    return y_test, y_pred, test_time


def run_classification(clf, cv, input_path):
    # lists where each split result is stored
    splits_cm = []
    splits_train_time = []
    splits_test_time = []

    # read input file as dataframe
    task = pd.read_csv(input_path, sep=";")

    # get range of all model features
    contiguous_columns = ["DurationTot", "Instruction"]
    features_range = np.r_[task.columns.get_loc(contiguous_columns[0]) : 
                           task.columns.get_loc(contiguous_columns[1]) + 1]
    # get model feature names and label name
    model_features = task.columns[features_range]
    model_label = task.columns[-1]

    # create two subframes containing only model features and model label
    X = task[model_features]
    y = task[model_label]
    # get unique namespace of possible labels
    labels_space = np.unique(y)

    # for each different split, test and train classifier
    for train_index, test_index in cv.split(X, y):
        trained_clf, training_time = train_classifier(clf, X, y, train_index)
        true_labels, predicted_labels, testing_time = test_classifier(clf, X, y, test_index)
        splits_cm.append(confusion_matrix(true_labels, predicted_labels))
        splits_train_time.append(training_time)
        splits_test_time.append(testing_time)

    return splits_cm, splits_train_time, splits_test_time
