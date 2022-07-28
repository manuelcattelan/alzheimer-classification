from src.utils.scan_input import scan_input_dir
from src.utils.preprocessing import normalize_data
from src.utils.parameters_tuning import tune_classifier
from src.utils.performance import compute_runs_report
from src.utils.performance import compute_clf_report
from src.utils.reports import export_runs_report
from src.utils.reports import export_clf_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import time


def train_classifier(clf, X, y, train_index):
    # Define training subframe for current split training index
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Train classifier and time training
    start = time.time()
    trained_clf = clf.fit(X_train, y_train)
    stop = time.time()

    train_time = stop - start

    return trained_clf, train_time


def test_classifier(clf, X, y, test_index):
    # Define testing subframe for current split testing index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Make predictions on test data and time testing
    start = time.time()
    y_pred = clf.predict(X_test)
    stop = time.time()

    test_time = stop - start

    return y_test, y_pred, test_time


def run_classification(clf, cv, df, n_splits):
    # Lists for holding each split results
    split_cm_list = []
    split_time_list = []
    # Dictionary for holding each run results
    run_results_dict = {}
    # Separate dataframe into two subframes:
    # X contains all feature columns except for the label column
    # y contains only the label column
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    # Initialize loop iterators
    split_iter = 1
    run_iter = 1
    # For each split/run from cross_validator split
    for train_index, test_index in cv.split(X, y):
        # Train classifier on training subframe
        (trained_clf,
         train_time) = train_classifier(clf, X, y, train_index)
        # Test classifier on testing subframe
        (true_labels,
         predicted_labels,
         test_time) = test_classifier(trained_clf, X, y, test_index)
        # Compute split results and append them in corresponding lists
        split_cm = confusion_matrix(true_labels, predicted_labels)
        split_cm_list.append(split_cm)
        split_time_list.append((train_time, test_time))
        # If current run is over (all splits were evaluated):
        # store splits results of current run into current run results
        # reset splits results lists
        # update loop iterators
        if split_iter == n_splits:
            run_results_dict[run_iter] = (split_cm_list, split_time_list)
            split_cm_list = []
            split_time_list = []
            split_iter = 1
            run_iter = run_iter + 1
        else:
            split_iter = split_iter + 1

    return run_results_dict


def file_classification(clf,
                        cv,
                        input_path,
                        output_path,
                        normalize_df,
                        tune_mode,
                        tune_iter,
                        tune_parameters,
                        tune_metric,
                        n_splits,
                        n_jobs):
    # Read data from input path into dataframe
    df = pd.read_csv(input_path, sep=";")
    if normalize_df:
        df = normalize_data(df_to_normalize=df)
    # If tune_mode is specified, run parameter tuning and get optimized clf
    if tune_mode:
        (clf,
         clf_best_params,
         tune_time) = tune_classifier(clf=clf,
                                      df=df,
                                      tune_mode=tune_mode,
                                      tune_iter=tune_iter,
                                      tune_parameters=tune_parameters,
                                      tune_metric=tune_metric,
                                      n_jobs=n_jobs)
    # If tune_mode is not specified, set clf_best_params to default params
    else:
        clf_best_params = clf.get_params()
        tune_time = None
    # Run classification on dataframe
    run_results_dict = run_classification(clf=clf,
                                          cv=cv,
                                          df=df,
                                          n_splits=n_splits)
    # Compute run report and export to csv file
    runs_report = compute_runs_report(run_results_dict=run_results_dict)
    export_runs_report(input=input_path,
                       runs_report=runs_report,
                       tune_mode=tune_mode,
                       tune_parameters=clf_best_params,
                       tune_time=tune_time,
                       output=output_path)
    # Compute classification report and export to png file
    clf_report = compute_clf_report(runs_report=runs_report)
    export_clf_report(input=input_path,
                      clf_report=clf_report,
                      output=output_path)


def dir_classification(clf,
                       cv,
                       input_path,
                       output_path,
                       normalize_df,
                       tune_mode,
                       tune_iter,
                       tune_parameters,
                       tune_metric,
                       n_splits,
                       n_jobs):
    # Recursively scan input path in order to:
    # build a list of all input paths to read
    # build a list of all corresponding output paths to write
    input_path_list, output_path_list = scan_input_dir(input_path=input_path,
                                                       output_path=output_path)
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
            file_classification(clf=clf,
                                cv=cv,
                                input_path=input_filepath,
                                output_path=output_filepath,
                                normalize_df=normalize_df,
                                tune_mode=tune_mode,
                                tune_iter=tune_iter,
                                tune_parameters=tune_parameters,
                                tune_metric=tune_metric,
                                n_splits=n_splits,
                                n_jobs=n_jobs)
