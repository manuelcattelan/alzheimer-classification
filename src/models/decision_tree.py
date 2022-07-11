from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
import os

# Subdirectory used for model graphs
MODELS_DIR = 'models/dt/'
# Subdirectory used for model performances
PERFORMANCE_DIR = 'performance/dt/'
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

# Train model on all task data and export trained model
def export_model(model, X, y, input_path, output_path, index):
    # Add model directory to initial output path
    output_path = output_path + MODELS_DIR
    # Build output path to export results
    built_output_path = build_output_path(input_path, output_path)
    # Modify output path to correctly export .png file
    built_output_path = Path(built_output_path)
    built_output_path = (built_output_path.with_suffix('')).with_suffix('.png')

    # Train model on entire task data
    trained_model = train_model(model, X, y, index)

    # Plot trained model and export its .png representation
    tree.plot_tree(model, filled=True, rounded=True)
    plt.savefig(built_output_path, bbox_inches="tight", dpi=400)
    plt.close()

# Export confusion matrix
def export_confusion_matrix(cm, cm_performance, labels, input_path, output_path):
    # Add performance directory to initial output path
    output_path = output_path + PERFORMANCE_DIR
    # Build output path to export results
    built_output_path = build_output_path(input_path, output_path)
    # Modify output path to correctly export .png file
    built_output_path = Path(built_output_path)
    built_output_path = (built_output_path.with_suffix('')).with_suffix('.png')
    
    # Build performance text
    task_accuracy = cm_performance[0]
    task_precision = cm_performance[1]
    task_recall = cm_performance[2]
    task_f1score = cm_performance[3]
    performance_text = "\n\nAccuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 Score: {:.1f}%".format(task_accuracy, task_precision, task_recall, task_f1score)

    # Convert binary labels to original string labels
    labels = [*map(({0: 'Sano', 1: 'Malato'}).get, labels)]

    # Build heatmap with confusion matrix
    ax = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    # Set x/y axis labels
    ax.set_xlabel('\nPredicted values' + performance_text)
    ax.set_ylabel('Actual values')
    # Export heatmap to output path
    plt.savefig(built_output_path, bbox_inches="tight", dpi=400) 
    plt.close()

# Run classification on file
def run_classification_on_file(model, cv, input_path, output_path, mode):
    # List containing confusion matrix for each testing split
    split_results = []
    
    # Read file and store it as dataframe
    df = pd.read_csv(input_path, sep=SEP)

    # Feature columns used for classification
    features = df.columns[MODEL_FEATURES]
    # Label column used for classification
    label = MODEL_LABEL
    # Subset of dataframe containing entries with feature for classification
    X = df[features]
    # Subset of dataframe containing entries with labels for classification
    y = df[label]
    # Get unique labels from set of possible labels
    labels = np.unique(y)

    # Classification start time
    start_time = time.time()

    # For each split, train and test model, then store results
    for train_index, test_index in cv.split(X, y):
        # Get trained model
        trained_model = train_model(model, X, y, train_index)
        # Get expected and obtained results from testing
        model_actual, model_predicted = test_model(trained_model, X, y, test_index)
        # Compute confusion matrix on predictions
        split_conf_matrix = confusion_matrix(model_actual, model_predicted, labels=labels)
        # Store split result in split results list
        split_results.append(split_conf_matrix)

    # Classification stop time
    stop_time = time.time()
    # Classification delta time 
    task_time = stop_time - start_time

    # Compute task confusion matrix and performance metrics
    task_conf_matrix = sum(matrix for matrix in split_results)
    tn, fp, fn, tp = task_conf_matrix.ravel()
    task_accuracy = ((tp + tn) / (tp + fp + fn + tn) * 100)
    task_precision = (tp / (tp + fp) * 100)
    task_recall = (tp / (tp + fn) * 100)
    task_f1score = 2 * task_precision * task_recall / (task_precision + task_recall)

    # Create tuple containing task performance
    task_performance = (task_accuracy, task_precision, task_recall, task_f1score)
    export_model(model, X, y, input_path, output_path, df.index)
    export_confusion_matrix(task_conf_matrix, task_performance, labels, input_path, output_path)

    # Return tuple containing task performance
    return task_performance, task_time

# Run classification on directory of files
def run_classification_on_dir(model, cv, input_path, output_path):
    # List of results for each task
    tasks_results = []
    # List of classification times for each task
    tasks_times = []

    # List of task paths for each task file inside input directory
    tasks = sorted(glob.glob(os.path.join(input_path, '*.csv'))) 

    # For each task in directory, run classification and store results
    for task in tasks:
        # Run classification on each file from input directory
        task_results, task_time = run_classification_on_file(model, cv, task, output_path, mode='dir')
        # Store results in results list
        tasks_results.append(task_results)
        # Store time in times list
        tasks_times.append(task_time)

    # Compute best task classification information
    best_task_metric = max([ results[BEST_TASK_METRIC] for results in tasks_results ])
    best_task_index = [ results[BEST_TASK_METRIC] for results in tasks_results ].index(best_task_metric)
    best_task_time = tasks_times[best_task_index]

    # Obtain best task performances
    best_task_accuracy = tasks_results[best_task_index][0] 
    best_task_precision = tasks_results[best_task_index][1] 
    best_task_recall = tasks_results[best_task_index][2] 
    best_task_f1score = tasks_results[best_task_index][3]

    # Classification information
    total_classification_time = sum([ time for time in tasks_times ])
    avg_classification_time = np.mean([ time for time in tasks_times ])
    classification_metric = {0: 'accuracy', 1: 'precision', 2: 'recall'}[BEST_TASK_METRIC]

    # Print classification classification results
    print("\nDT classification on {} took: {:.3f}s (avg: {:.3f}s)"
            .format(input_path, total_classification_time, avg_classification_time))
    print("Best performing task (considering {}) was T{}, with the following results:"
            .format(classification_metric, best_task_index + 1))
    print("Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 Score: {:.1f}%\nTime: {:.3f}s"
            .format(best_task_accuracy, best_task_precision, best_task_recall, best_task_f1score, best_task_time))

# Helper function to build correct output path for an input file
def build_output_path(input_path, output_path):
    # Extract input source (last folder before filename) and filename from input path
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
    # -o flag defines output path to where classification results are stored and is necessary
    parser.add_argument('-o', type=str, metavar='<output_folder>', help="output directory where built data is saved", required=True)

    # Parse cli arguments and store them in variable 'args'
    args = parser.parse_args()

    # Store -o flag and check its validity
    output_path = args.o
    # Get output path extension (empty if not present)
    output_path_extension = (os.path.splitext(output_path))[1]
    # If output path has an extension, throw error because it does not define a dir path
    if output_path_extension != '':
        raise ValueError('specified output is not a directory')

    # If file flag is set
    if args.f:
        # Store -f flag
        input_path = args.f
        # Check if given argument is valid (input is file)
        input_is_file = os.path.isfile(input_path)
        # If input argument is valid, run classification
        if (input_is_file):
            # Initialize classifier
            decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
            # Initialize cross validator
            cross_validator = StratifiedKFold(n_splits=SPLITS, shuffle=True)
            # Run classification on single file
            run_classification_on_file(decision_tree, cross_validator, input_path, output_path, mode='file')
        # If input is not a file
        else:
            # Raise exception
            raise ValueError('specified input is not a file')

    # if directory flag is set
    if args.d:
        # Store -d flag
        input_path = args.d
        # Check if given argument is valid (input is directory)
        input_is_dir = os.path.isdir(input_path)
        # If input argument is valid, run classification 
        if (input_is_dir):
            # Initialize classifier
            decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
            # Initialize cross validator
            cross_validator = StratifiedKFold(n_splits=SPLITS, shuffle=True)
            # Run classification on all files inside directory
            run_classification_on_dir(decision_tree, cross_validator, input_path, output_path)
        # If input is not a directory
        else:
            # Raise exception
            raise ValueError('specified input is not a directory')

# Main loop
if __name__ == "__main__":
    main()
