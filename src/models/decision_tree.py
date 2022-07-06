from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from pathlib import Path
import time
import numpy as np
import pandas as pd
import argparse
import glob
import os

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
    model.fit(X_train, y_train)

    # Return trained model
    return model

# Test classifier and return accuracy result
def test_model(model, X, y, test_index):
    # Define testing dataframes for current split index
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Model prediction on test data
    y_pred = model.predict(X_test)
    # Compute confusion matrix on prediction
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Return model accuracy
    return conf_matrix

# Run classification on dataframe
def run_classification_on_df(model, cv, input, output):
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

    # Classification start time
    start_time = time.time()

    # For each split, train and test model, then store results
    for train_index, test_index in cv.split(X, y):
        trained_model = train_model(model, X, y, train_index)
        split_conf_matrix = test_model(model, X, y, test_index)
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

    # Write results to corresponding task file
    with open(output, 'w') as f:
        print("Results for '{}'".format(input), file=f)
        print("Confusion matrix\n{}". format(task_conf_matrix), file=f)
        print("Accuracy: {:.1f}%".format(task_accuracy), file=f)
        print("Precision: {:.1f}%".format(task_precision), file=f)
        print("Recall: {:.1f}%".format(task_recall), file=f)
        print("Time: {:.3f}s".format(task_time), file=f)
        f.close()

# Run classification on df list
def run_classification_on_ds(model, cv, input_path, output_path):
    # List of names for each file inside input dir
    input_paths = sorted(glob.glob(os.path.join(input_path, '*.csv'))) 
    input_file_names = [ os.path.basename(input_path) for input_path in input_paths]
    csv_output_paths = [ os.path.join(output_path, file_name) for file_name in input_file_names ]
    txt_output_paths = [ (os.path.splitext(path)[0] + '.txt') for path in csv_output_paths ]
    # Read every file inside folder and store it in list as df
    ds = [ pd.read_csv(df, sep=SEP) for df in input_paths]

    # List of results for each task
    tasks_results = []

    # For each df in list, run classification on it and print results to corresponding output file
    for input, output in zip(input_paths, txt_output_paths):
        run_classification_on_df(model, cv, input, output)

def main():
    # Set up parser
    parser = argparse.ArgumentParser(description="Decision tree classifier",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add possible cli arguments to parser
    parser.add_argument("-i", help="Input to read in .csv format")
    parser.add_argument("-o", help="Output to write in .csv format")
    # Parse cli arguments and store them in variable 'args'
    args = parser.parse_args()
    args = vars(args)
    # Store cli arguments
    input_path = args['i']
    output_path = args['o']

    # Create output directory if it does not exist
    output_file_dir = Path(os.path.dirname(output_path))
    output_file_dir.mkdir(parents=True, exist_ok=True)
    
    # Check whether input and output arguments are file or directory
    input_is_file = os.path.isfile(input_path)
    input_is_dir = os.path.isdir(input_path)
    output_is_dir = os.path.isdir(output_path)

    # Initialize classifier
    decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
    # Initialize cross validator
    cross_validator = StratifiedKFold(n_splits=SPLITS, shuffle=True)

    # If input argument is a file, only run classification on that file
    if (input_is_file and not(output_is_dir)):
        # Run classification on single file
        run_classification_on_df(decision_tree, cross_validator, input_path, output_path)
            
    # If input argument is a folder, run classification on every file inside the folder
    if (input_is_dir and output_is_dir):
        # Run classification on all files inside directory
        run_classification_on_ds(decision_tree, cross_validator, input_path, output_path)

if __name__ == "__main__":
    main()
