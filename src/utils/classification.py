from src.utils.scan_input import scan_input_dir
from src.utils.parameters_tuning import tune_classifier
from src.utils.performance import compute_classifier_performance
from src.utils.performance import compute_best_task_performance
from src.utils.reports import print_file_classification_report
from src.utils.reports import print_dir_classification_report
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


def run_classification(clf, cv, df):
    # lists where each split result is stored
    splits_cm = []
    splits_train_time = []
    splits_test_time = []

    # create two subframes containing only model features and model label
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # get unique namespace of possible labels
    labels_space = np.unique(y)

    # for each different split
    for train_index, test_index in cv.split(X, y):
        # train classifier
        (trained_clf,
         training_time) = train_classifier(clf, X, y, train_index)
        # test classifier
        (true_labels,
         predicted_labels,
         testing_time) = test_classifier(clf, X, y, test_index)
        # store results
        splits_cm.append(confusion_matrix(true_labels, predicted_labels))
        splits_train_time.append(training_time)
        splits_test_time.append(testing_time)

    return splits_cm, splits_train_time, splits_test_time


def file_classification(
        clf,
        cv,
        input_path,
        output_path,
        tune_params,
        tune_arg):
    # read input file as dataframe
    df = pd.read_csv(input_path, sep=";")
    # run parameters tuning if tune argument is not None
    if tune_arg:
        clf, tuning_best_params, tuning_time = tune_classifier(
                clf, cv, tune_params, df, tune_arg
                )
    # run classification
    splits_cm, splits_train_time, splits_test_time = run_classification(
            clf, cv, df
            )
    # compute classifier performance
    (task_performance,
     task_train_time,
     task_test_time) = compute_classifier_performance(
             splits_cm, splits_train_time, splits_test_time
             )

    return task_performance, task_train_time, task_test_time


def dir_classification(
        clf,
        cv,
        input_path,
        output_path,
        tune_params,
        tune_arg,
        metric_arg):
    # get input file path and build corresponding output file path
    # of all files inside input directory
    input_paths, output_paths = scan_input_dir(input_path, output_path)
    # for each directory found while traversing input dir
    for input_dirpath, output_dirpath in zip(
            sorted(input_paths),
            sorted(output_paths)):
        # list of paths of files inside currently considered dir
        input_filepaths = sorted(input_paths[input_dirpath])
        output_filepaths = sorted(output_paths[output_dirpath])
        # if there's only one file inside the current dir,
        if len(input_filepaths) == 1:
            # run single file classification
            file_classification(
                    clf,
                    cv,
                    input_filepaths[0],
                    output_filepaths[0],
                    tune_params,
                    tune_arg
                    )
        # if there's at least two files inside
        # current directory, run dir classification
        else:
            # lists holding every task's classification information
            tasks_performance = []
            tasks_train_time = []
            tasks_test_time = []
            tasks_best_params = []
            tasks_tuning_times = []

            # for each file inside currently considered dir
            for input_filepath, output_filepath in zip(
                    input_filepaths,
                    output_filepaths
                    ):
                # run classification process on file
                (task_performance,
                 task_train_time,
                 task_test_time) = file_classification(
                         clf,
                         cv,
                         input_filepath,
                         output_filepath,
                         tune_params,
                         tune_arg
                         )
                # store classification results
                tasks_performance.append(task_performance)
                tasks_train_time.append(task_train_time)
                tasks_test_time.append(task_test_time)

            # compute best task
            (best_task_performance,
             best_task_train_time,
             best_task_test_time,
             best_task_index) = compute_best_task_performance(
                tasks_performance,
                tasks_train_time,
                tasks_test_time,
                metric_arg
                )

            # compute classification total time and per avg time per task
            total_train_time = sum(time for time in tasks_train_time)
            total_test_time = sum(time for time in tasks_test_time)
            avg_train_time = np.mean([time for time in tasks_train_time])
            avg_test_time = np.mean([time for time in tasks_test_time])
