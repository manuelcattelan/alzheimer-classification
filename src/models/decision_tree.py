from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
import time
import numpy as np
import pandas as pd
import argparse
import glob
import os

# Default results dir
DEFAULT_OUTPUT_DIR = "results/"
# Metric used to evaluate best task (0: accuracy, 1: precision, 2: recall)
BEST_TASK_METRIC = 0
# Features range used by the model to make classification
MODEL_FEATURES= np.r_[ 1:91 ]
# Feature name used by the model as the classification label
MODEL_LABEL = 'Label'
# Number of splits used in stratified k-fold
SPLITS = 10
# Character used for reading .csv files
SEP = ','

# Train model and return trained model
def train_model(model, X, y, train_index):
    # Define training dataframes for current split index
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Train model on train data
    trained_model = model.fit(X_train, y_train)

    # Return trained model
    return trained_model

# Test classifier and return accuracy result
def test_model(model, X, y, test_index):
    # Define testing dataframes for current split index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Model prediction on test data
    y_pred = model.predict(X_test)

    # Return model test data and corresponding predictions
    return y_test, y_pred

# Run classification on dataframe
def run_classification_on_df(model, cv, input, output, mode):
    # Read file and store it as df
    df = pd.read_csv(input, sep=SEP)

    # List containing confusion matrix for each testing split
    split_conf_matrices = []

    # List of features for classification
    features = df.columns[MODEL_FEATURES]
    # Label used for classification
    label = MODEL_LABEL

    # Subset of dataframe containing entries with feature for classification
    X = df[features]
    # Subset of dataframe containing entries with labels for classification
    y = df[label]
    # Get unique labels from classification labels
    labels = np.unique(y)

    # Classification start time
    start_time = time.time()

    # For each split, train and test model, then store results
    for train_index, test_index in cv.split(X, y):
        trained_model = train_model(model, X, y, train_index)
        model_actual, model_predicted = test_model(trained_model, X, y, test_index)

        # Compute confusion matrix on prediction and append it to splits matrices list
        split_conf_matrix = confusion_matrix(model_actual, model_predicted, labels=labels)
        split_conf_matrices.append(split_conf_matrix)

    # Classification stop time
    stop_time = time.time()
    # Classification delta time 
    task_time = stop_time - start_time

    # Compute final confusion matrix
    task_conf_matrix = sum(matrix for matrix in split_conf_matrices)
    tn, fp, fn, tp = task_conf_matrix.ravel()

    # Compute performance metrics from confusion matrix
    task_accuracy = ((tp + tn) / (tp + fp + fn + tp) * 100)
    task_precision = (tp / (tp + fp) * 100)
    task_recall = (tp / (tp + fn) * 100)

    # Convert binary labels to original string labels
    labels = [*map(({0: 'Sano', 1: 'Malato'}).get, labels)]

    # Return tuple containing task performance
    return (task_accuracy, task_precision, task_recall)

# Run classification on df list
def run_classification_on_ds(model, cv, input_path, output_path):
    # List of names for each file inside input dir
    input_paths = sorted(glob.glob(os.path.join(input_path, '*.csv'))) 
    output_paths = [ build_output_path(input_path, output_path) for input_path in input_paths]

    # List of results for each task
    tasks_results = []

    # For each df in list, run classification on it and print results to corresponding output file
    for input, output in zip(input_paths, output_paths):
        task_results = run_classification_on_df(model, cv, input, output, mode='dir')
        tasks_results.append(task_results)

    # Get best performing task based on defined metric
    best_task = max([metrics[BEST_TASK_METRIC] for metrics in tasks_results])
    best_task_index = [metrics[BEST_TASK_METRIC] for metrics in tasks_results].index(best_task)
    task_metric = {0: 'accuracy', 1: 'precision', 2: 'recall'}[BEST_TASK_METRIC]
    # Print best performing task info
    print("Best performing task for {} was T{} with {:.1f}% {}".format(input_path,
                                                                       best_task_index,
                                                                       best_task,
                                                                       task_metric))

# Helper function to build correct output string for an input file
def build_output_path(input_path, output_path):
    # Get path extension if present
    path_extension = (os.path.splitext(output_path))[1]
    # If path has an extension, throw error because it does not describe a dir path
    if path_extension != '':
        raise ValueError('specified output is not a directory')

    # Extract input source (last folder before filename) and filename
    input_source = os.path.basename(os.path.dirname(input_path))
    input_filename = os.path.basename(input_path)

    # Create output directory with input source if it does not exist
    output_dir = Path(os.path.dirname(output_path))
    output_dir = output_dir / input_source
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output path by joining new output directory with input filename
    output_path = output_dir / input_filename

    # Return final output path
    return output_path

# Main function
def main():
    # Set up parser
    parser = argparse.ArgumentParser(prog="decision_tree.py",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add possible cli arguments to parser
    action = parser.add_mutually_exclusive_group(required=True)
    # -f flag and -d flag are mutually exclusive and necessary
    action.add_argument('-f', type=str, metavar='<input_file>', help="input .csv file to build")
    action.add_argument('-d', type=str, metavar='<input_source>', help="input directory from which to take .csv <input_file>s to build")
    # -o flag is optional and default value is declared as constant
    parser.add_argument('-o', type=str, metavar='<output_folder>', help="output directory where built data is saved (default is /data/processed/<input_source>/)")

    # Parse cli arguments and store them in variable 'args'
    args = parser.parse_args()

    # Store cli arguments
    input_file = args.f
    input_dir = args.d
    # If output flag is defined, use it's argument as output_dir, else use default constant
    output_dir = args.o if args.o else DEFAULT_OUTPUT_DIR

    # If file flag is set
    if input_file:
        # Check if given arguments are valid (input is file and output is dir)
        input_is_file = os.path.isfile(input_file)

        # If conditions are met build data
        if (input_is_file):
            # Initialize classifier
            decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
            # Initialize cross validator
            cross_validator = StratifiedKFold(n_splits=SPLITS, shuffle=True)
            # Build output path
            output_path = build_output_path(input_file, output_dir)
            # Run classification on single file
            run_classification_on_df(decision_tree, cross_validator, input_file, output_path, mode='file')
        # If input is not a directory
        else:
            # Raise exception
            raise ValueError('specified input is not a file')

    # if directory flag is set
    if input_dir:
        # Check if given arguments are valid (input is dir and output is dir)
        input_is_dir = os.path.isdir(args.d)
        
        # If conditions are met build data
        if (input_is_dir):
            # Initialize classifier
            decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
            # Initialize cross validator
            cross_validator = StratifiedKFold(n_splits=SPLITS, shuffle=True)
            # Run classification on all files inside directory
            run_classification_on_ds(decision_tree, cross_validator, input_dir, output_dir)
        # If input is not a directory
        else:
            # Raise exception
            raise ValueError('specified input is not a directory')

# Main loop
if __name__ == "__main__":
    main()
