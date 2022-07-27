from src.utils.scan_input import scan_input_dir
from src.utils.preprocessing import normalize_data
from src.utils.parameters_tuning import tune_classifier
from src.utils.performance import compute_runs_report
from src.utils.performance import compute_classification_report
from src.utils.reports import print_runs_report
from src.utils.reports import export_classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import time


def train_classifier(classifier, X, y, train_index):
    # Define training subframe for current split training index
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Train classifier and time training
    start = time.time()
    trained_clf = classifier.fit(X_train, y_train)
    stop = time.time()

    train_time = stop - start

    return trained_clf, train_time


def test_classifier(classifier, X, y, test_index):
    # Define testing subframe for current split testing index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Make predictions on test data and time testing
    start = time.time()
    y_pred = classifier.predict(X_test)
    stop = time.time()

    test_time = stop - start

    return y_test, y_pred, test_time


def run_classification(classifier, cross_validator, df, splits):
    # Lists for holding each split results
    split_confusion_matrix_list = []
    split_times_list = []
    # Dictionary for holding each run results
    run_results_dict = {}
    # Separate dataframe into two subframes:
    # X contains all feature columns except for the label column
    # y contains only the label column
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    # Unique namespace of possible labels from dataframe
    labels_space = np.unique(y)

    # Initialize loop iterators
    split_iter = 1
    run_iter = 1
    # For each split/run from cross_validator split
    for train_index, test_index in cross_validator.split(X, y):
        # Train classifier on training subframe
        trained_clf, train_time = train_classifier(classifier,
                                                   X,
                                                   y,
                                                   train_index)
        # Test classifier on testing subframe
        true_labels, predicted_labels, test_time = test_classifier(classifier,
                                                                   X,
                                                                   y,
                                                                   test_index)
        # Compute split results and append them in corresponding lists
        split_confusion_matrix = confusion_matrix(true_labels,
                                                  predicted_labels)
        split_confusion_matrix_list.append(split_confusion_matrix)
        split_times_list.append((train_time, test_time))
        # If current run is over (all splits were evaluated)
        if split_iter == splits:
            # Store splits results inside current run results
            run_results_dict[run_iter] = (split_confusion_matrix_list,
                                          split_times_list)
            # Reset split results lists
            split_confusion_matrix_list = []
            split_times_list = []
            # Update loop iterators
            split_iter = 1
            run_iter = run_iter + 1
        else:
            split_iter = split_iter + 1

    return run_results_dict


def file_classification(classifier,
                        cross_validator,
                        input_path,
                        output_path,
                        normalize,
                        jobs,
                        tune_mode,
                        tune_iter,
                        tune_parameters,
                        splits):
    # Read data from input path into dataframe
    df = pd.read_csv(input_path, sep=";")
    if normalize:
        df = normalize_data(df)
    # If tune_mode is specified, run parameter tuning
    if tune_mode:
        (classifier,
         classifier_best_parameters,
         tune_time) = tune_classifier(classifier,
                                      cross_validator,
                                      df,
                                      jobs,
                                      tune_mode,
                                      tune_iter,
                                      tune_parameters)
    # Run classification on dataframe
    run_results_dict = run_classification(classifier,
                                          cross_validator,
                                          df,
                                          splits)
    # Compute classification report
    runs_report = compute_runs_report(run_results_dict)
    print_runs_report(input_path,
                      runs_report,
                      tune_mode,
                      classifier_best_parameters,
                      tune_time)
    classification_report = compute_classification_report(runs_report)
    export_classification_report(input_path,
                                 classification_report,
                                 output_path)


def dir_classification(classifier,
                       cross_validator,
                       input_path,
                       output_path,
                       normalize,
                       jobs,
                       tune_mode,
                       tune_iter,
                       tune_parameters,
                       splits,
                       metric):
    # Recursively scan input path in order to:
    # build a list of all input paths to read
    # build a list of all corresponding output paths to write
    input_path_list, output_path_list = scan_input_dir(input_path, output_path)
    # For each directory path found while traversing input path
    for input_dirpath, output_dirpath in zip(sorted(input_path_list),
                                             sorted(output_path_list)):
        # List of paths of files inside currently considered dir
        input_filepath_list = sorted(input_path_list[input_dirpath])
        output_filepath_list = sorted(output_path_list[output_dirpath])
        # for each file inside currently considered dir
        for input_filepath, output_filepath in zip(input_filepath_list,
                                                   output_filepath_list):
            # run classification process on file
            file_classification(classifier,
                                cross_validator,
                                input_filepath,
                                output_filepath,
                                normalize,
                                jobs,
                                tune_mode,
                                tune_iter,
                                tune_parameters,
                                splits)
