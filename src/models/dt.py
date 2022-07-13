from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os

def init_clf(input_path):
    # read input file as dataframe
    task = pd.read_csv(input_path, sep=';')

    # get range of all model features
    contiguous_columns = ['DurationTot', 'Instruction']
    features_range = np.r_[task.columns.get_loc(contiguous_columns[0]):
                           task.columns.get_loc(contiguous_columns[1]) + 1]
    # get model feature names and label name
    model_features = task.columns[features_range]
    model_label = task.columns[-1]

    # create two subframes containing only model features and model label
    X = task[model_features]
    y = task[model_label]

    return X, y

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

    # get dataframes used for classification
    X, y = init_clf(input_path)
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
    export_clf_performance(task_cm, task_performance, labels_space, input_path, output_path)

    return task_performance, task_time

def compute_clf_performance(cm):
    # get each matrix cell separately
    tn, fp, fn, tp = cm.ravel()

    # compute task performance
    accuracy = ((tp + tn) / (tn + fp + fn + tp) * 100)
    precision = (tp / (tp + fp) * 100)
    recall = (tp / (tp + fn) * 100)
    f1_score = (2 * precision * recall / (precision + recall))

    return (accuracy, precision, recall, f1_score)

def compute_clf_best_task(tasks_results, tasks_times, p_metric):
    # map metric argument to corresponging index in performance tuple
    metric = {'accuracy': 0, 'precision': 1, 'recall': 2, 'f1': 3}[p_metric]

    # Compute best task classification information
    best_task_metric = max([ results[metric] for results in tasks_results ])
    best_task_index = [ results[metric] for results in tasks_results ].index(best_task_metric)
    best_task_time = tasks_times[best_task_index]

    # Obtain best task performances
    best_task_accuracy = tasks_results[best_task_index][0] 
    best_task_precision = tasks_results[best_task_index][1] 
    best_task_recall = tasks_results[best_task_index][2] 
    best_task_f1score = tasks_results[best_task_index][3]

    return (best_task_accuracy, best_task_precision, best_task_recall, best_task_f1score), best_task_time, best_task_index

def build_output_path(input_path, output_path):
    # extract input parent dir and filename from input path
    input_parent_dir = os.path.basename(os.path.dirname(input_path))
    input_filename = os.path.basename(input_path)

    # join initial output path with input parent dir 
    # (e.g. add air/ to data/processed/)
    # and create result directory if it does not exist
    output_dirpath = output_path
    output_dirpath = output_dirpath / input_parent_dir
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # join new output path with input file name
    # this is the path to which the current file will be saved
    output_filepath = output_dirpath / input_filename

    return output_filepath

def export_clf_performance(cm, performance, labels, input_path, output_path):
    # build output path corresponding to input filename being classified
    output_path = output_path / 'performance/dt/'
    output_path = build_output_path(input_path, output_path)
    output_path = (output_path.with_suffix('')).with_suffix('.png')

    # map boolean labels to strings
    labels = [*map(({0: 'Sano', 1: 'Malato'}).get, labels)]
    # build performance text to display under cm heatmap
    performance_text = '\n\nAccuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 score: {:.1f}%'.format(performance[0],
                                                                                                              performance[1],
                                                                                                              performance[2],
                                                                                                              performance[3])

    # Build heatmap with confusion matrix
    ax = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    ax.set_xlabel('\nPredicted values' + performance_text)
    ax.set_ylabel('Actual values')

    # Export heatmap to output path
    plt.savefig(output_path, bbox_inches="tight", dpi=400) 
    plt.close()

def main():
    # set up parser and possible arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i',
                        type=str,
                        metavar='<input_file/dir>',
                        help='input file or directory to classify',
                        required=True)
    parser.add_argument('-s',
                        type=int,
                        metavar='<n_splits>',
                        help='number of splits used for k-fold cross validation',
                        default=10)
    parser.add_argument('-m',
                        type=str,
                        metavar='<p_metric>',
                        help='metric used to determine best performing task (when using dir classification)',
                        default='accuracy')
    parser.add_argument('-o',
                        type=str,
                        metavar='<output_dir>',
                        help='output directory where results are stored',
                        required=True)
    args = parser.parse_args()
    args = vars(args)

    # store parsed arguments
    input_path = Path(args['i'])
    output_path = Path(args['o'])

    # check output argument validity by checking
    # if it ends with any extension
    output_path_extension = (os.path.splitext(output_path))[1]
    if output_path_extension != '':
        raise ValueError(output_path + ' is not a valid directory path')

    # check inputfile argument validity by checking
    # if it points to an existing file
    if (os.path.isfile(input_path)):
        # define classifier and cross validator
        clf = tree.DecisionTreeClassifier()
        cv = StratifiedKFold(n_splits=args['s'], shuffle=True)
        # run classification on file
        results, time = run_clf(clf, cv, input_path, output_path) 
        print('Classification on {} took {:.3f}s:'
              .format(input_path, time))
        print('Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 score: {:.1f}%'
              .format(results[0], results[1], results[2], results[3]))

    # check inputdir argument validity by checking
    # if it points to an existing directory
    elif (os.path.isdir(input_path)):
        # define classifier and cross validator
        clf = tree.DecisionTreeClassifier()
        cv = StratifiedKFold(n_splits=args['s'], shuffle=True)
        # traverse input directory and find all .csv files
        input_dict = {}
        for root, dirs, files in os.walk(input_path):
            # if files were found in currently walked dir
            if files:
                csv_files = [] 
                for file in files:
                    if (os.path.splitext(file))[1] == '.csv':
                        csv_files.append(file)
                input_dict[root] = sorted(csv_files)

        # for each dir inside input argument, make classification on all files inside of it  
        for dir in input_dict:
            # list of files inside dir
            input_filepaths = input_dict[dir]
            # list where each task result is stored
            input_results = []
            # list where each task time is stored
            input_times = []
            # run classification on each file inside input dir
            for input_filepath in input_filepaths:
                results, time = run_clf(clf, cv, input_filepath, output_path)
                input_results.append(results)
                input_times.append(time)

            results, time, index = compute_clf_best_task(input_results, input_times, args['m'])
            total_clf_time = sum([ time for time in input_times ])
            avg_clf_time = np.mean([ time for time in input_times ])

            print('Classification on {} took: {:.3f}s (avg: {:.3f}s)'
                  .format(dir, total_clf_time, avg_clf_time))
            print('Best performing task (wrt {}) was T{}, with the following results:'
                  .format(args['m'], index + 1))
            print('Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 Score: {:.1f}%\nTime: {:.3f}s'
                  .format(results[0], results[1], results[2], results[3], time))
    else:
        raise ValueError(input_path / ' is neither an existing file nor directory')

if __name__ == '__main__':
    main()
