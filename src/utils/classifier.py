from src.utils.preprocessing import *
from src.utils.performance import compute_clf_performance
from sklearn.metrics import confusion_matrix
import time
import pandas as pd
import numpy as np

def init_clf(task):
    # get range of all model features
    contiguous_columns = ['DurationTot', 'Instruction']
    features_range = np.r_[task.columns.get_loc(contiguous_columns[0]):
                           task.columns.get_loc(contiguous_columns[1]) + 1]
    # get model feature names and label name
    model_features = task.columns[features_range]
    model_label = task.columns[-1]

    return model_features, model_label

def train_clf(clf, X, y, train_index):
    # define training subframe for current split training index
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # train classifier
    trained_clf = clf.fit(X_train, y_train)

    return trained_clf

def test_clf(clf, X, y, test_index):
    # define testing subframe for current split testing index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # make predictions on test data
    y_pred = clf.predict(X_test)

    return y_test, y_pred

def run_clf(clf, cv, input_path, output_path):
    # list where each split confusion matrix is stored
    splits_cm = []
    # read input file as dataframe
    task = build_data(input_path, output_path)

    # get dataframes used for classification
    features, label = init_clf(task)

    # create two subframes containing only model features and model label
    X = task[features]
    y = task[label]
    # get unique namespace of possible labels
    labels_space = np.unique(y)

    # time classification for each task (time for all splits to complete)
    start = time.time()
    # for each different split, test and train classifier
    for train_index, test_index in cv.split(X, y):
        trained_clf = train_clf(clf, X, y, train_index)
        true_labels, predicted_labels = test_clf(clf, X, y, test_index)
        splits_cm.append(confusion_matrix(true_labels, predicted_labels))

    # get task final classification time
    stop = time.time()
    task_time = (stop - start)

    # compute final task confusion matrix and its metrics
    task_cm = sum(cm for cm in splits_cm)
    task_performance = compute_clf_performance(task_cm)
    # export task results to file
    # export_clf_performance(task_cm, task_performance, labels_space, output_path)
    # export_clf_representation(clf, X, y, features, labels_space, output_path, task_index_list)

    return task_performance, task_time
