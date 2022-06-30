# Modules
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import glob
import os

# Folder path for processed input data
AIR_PROCESSED_DATAPATH = "../../data/processed/air/"
PAPER_PROCESSED_DATAPATH = "../../data/processed/paper/"
AP_PROCESSED_DATAPATH = "../../data/processed/ap/"
# Folder path for matrices output
AIR_RESULTS_DATAPATH = "../../results/figures/air" 
PAPER_RESULTS_DATAPATH = "../../results/figures/paper"
AP_RESULTS_DATAPATH = "../../results/figures/ap"

# Column range for features
FEATURES_RANGE = np.r_[ 1:91 ]
# Separator character used to read input data
SEP = "," 

# Number of folds for stratified cross validation
SPLITS = 10

# Train classifier and return trained classifier
def train_classifier(classifier, X, y, train_index):
    # Training dataframe for current split
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Train classifier on train data
    classifier.fit(X_train, y_train)

    # Return trained classifier
    return classifier

# Test classifier and return obtained score as mean accuracy
def test_classifier(classifier, X, y, test_index):
    # Testing dataframe for current split
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Make prediction on test data
    y_pred = classifier.predict(X_test)
    # Compute accuracy on prediction
    classifier_accuracy = accuracy_score(y_test, y_pred)
    # Compute confusion matrix on prediction
    classifier_conf_matrix = confusion_matrix(y_test, y_pred)

    # Return accuracy and confusion matrix of classification
    return classifier_accuracy, classifier_conf_matrix

# Print best results for each dataframe list (eg. best performing task for air input)
def get_best_task(task_results):
    # Initialize average result variables
    best_task_value = 0.0
    best_task_index = 0.0

    # For each result in the result array (list of split results)
    for new_task_index, new_task_result in enumerate(task_results):
        # If new average is greater than old average, save it as new max
        if (new_task_result >= best_task_value):
            best_task_value = new_task_result
            best_task_index = new_task_index

    # Return best task accuracy and number
    return best_task_value, best_task_index

# Run classification on dataframe
def run_classification_per_df(classifier, cv, df, task_index, ds_type):
    # List used to store all split results for a single task
    split_results = []
    split_matrices = []

    # Extract features to consider from dataframe 
    features_column = df.columns[FEATURES_RANGE]
    # Name of classification column
    label_column = 'Label'

    # Subset of dataframe containing entries with features to use for classification
    X = df[features_column]
    # Subset of dataframe containing entries with labels to use for classification
    y = df[label_column] 

    # Classification start time
    start_time = time.time()

    # For each split, train and test classifier, then store result split result/matrix
    for train_index, test_index in cv.split(X, y):
        trained_classifier = train_classifier(classifier, X, y, train_index)
        split_result, split_conf_matrix = test_classifier(trained_classifier, X, y, test_index)
        split_results.append(split_result)
        split_matrices.append(split_conf_matrix)

    # Classification end time
    end_time = time.time()

    # Elapsed classification time
    task_time = end_time - start_time
    # Compute task result as the mean of all split results for the particular task
    task_result = np.mean(split_results)
    # Compute task confusion matrix as the sum of all split confusion matrices
    task_matrix = sum(matrix for matrix in split_matrices)

    # Export confusion matrix for current task
    export_confusion_matrix(task_matrix, ds_type, task_index)

    # Return classifier results
    return task_result, task_matrix, task_time

# Run classification on dataset
def run_classification_per_ds(classifier, cv, ds, ds_type):
    # List of results for each task
    ds_results = []
    # List of confusion matrices for each task
    ds_matrices = [] 
    # Keep track of classification times for each task
    ds_times = []

    # Run classification for every df in ds
    for df_index, df in enumerate(ds):
        task_result, task_matrix, task_time = run_classification_per_df(classifier, cv, df, df_index, ds_type)
        ds_results.append(task_result)
        ds_matrices.append(task_matrix)
        ds_times.append(task_time)

    # Get best performing task in ds
    best_task_value = max(ds_results)
    best_task_index = ds_results.index(best_task_value)
    best_task_matrix = ds_matrices[best_task_index]

    # Print classification times
    print("Total classification time: {:.3f}s".format(sum(ds_times)))
    print("Average classification time per task: {:.3f}s".format(np.mean(ds_times)))

    # Print classification results
    print("Task T{} provided the best results:".format(best_task_index + 1))

    # Compute and print performance metrics from confusion matrix
    tn, fp, fn, tp = best_task_matrix.ravel() 
    print("Accuracy: {:.1f}%".format((tp + tn) / (tp + tn + fp + fn) * 100))
    print("Precision: {:.1f}%".format(tp / (tp + fp) * 100))
    print("Recall: {:.1f}%".format(tp / (tp + fn) * 100))

    # Compute tree training parameters on best performing task
    feature_columns = ds[best_task_index].columns[FEATURES_RANGE]
    label_column = 'Label'
    X = ds[best_task_index][feature_columns]
    y = ds[best_task_index][label_column]
    train_index = ds[best_task_index].index

    # Train tree on best performing task
    best_task_tree = train_classifier(classifier, X, y, train_index)

    # Export best task tree as png
    export_tree(best_task_tree, ds, ds_type, best_task_index, feature_columns)

# Train and export final tree on best task for each dataset
def export_tree(classifier, ds, ds_type, task_index, feature_names):
    # Match filename of tree graph with current dataset
    match ds_type:
        case 'air':
            base_name = 'tree_air_T'
            base_folder_path = Path(AIR_RESULTS_DATAPATH)
        case 'paper':
            base_name = 'tree_paper_T'
            base_folder_path = Path(PAPER_RESULTS_DATAPATH)
        case 'ap':
            base_name = 'tree_ap_T'
            base_folder_path = Path(AP_RESULTS_DATAPATH)

    # Create directory in which to export data if it does not exist already
    base_folder_path.mkdir(parents = True, exist_ok = True)
    # Create filename using file index given as argument to match format of input data
    if (task_index < 9):
        file_no = '0' + str(task_index + 1)
    else:
        file_no = str(task_index + 1)

    # Create full file name and path
    full_file_name = base_name + file_no + '.png'
    full_file_path = base_folder_path / full_file_name

    # Plot and export tree graph for trained classifier
    tree.plot_tree(classifier, feature_names = feature_names, class_names = ['Sano', 'Malato'], filled = True) 
    plt.savefig(full_file_path, bbox_inches = "tight", dpi = 1200)
    plt.close()

# Export confusion matrix as png using heatmap
def export_confusion_matrix(cm, df_type, df_index):
    # Create heatmap from confusion matrix
    ax = sns.heatmap(cm, annot = True, cmap = 'Blues')

    # Plot options
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('\nPredicted values')
    ax.set_ylabel('Actual values')

    # Match filename of tree graph with current dataset
    match df_type:
        case 'air':
            base_name = 'onAir_T'
            base_folder_path = Path(AIR_RESULTS_DATAPATH)
        case 'paper':
            base_name = 'onPaper_T'
            base_folder_path = Path(PAPER_RESULTS_DATAPATH)
        case 'ap':
            base_name = 'onAirOnPaper_T'
            base_folder_path = Path(AP_RESULTS_DATAPATH)

    # Create directory in which to export data if it does not exist already
    base_folder_path.mkdir(parents = True, exist_ok = True)

    # Create filename using file index given as argument to match format of input data
    if (df_index < 9):
        file_no = '0' + str(df_index + 1)
    else:
        file_no = str(df_index + 1)

    # Create full file name and path
    full_file_name = base_name + file_no + '.png'
    full_file_path = base_folder_path / full_file_name

    # Export plot as image and close it
    plt.savefig(full_file_path)
    plt.close()

# Main function
def main():
    # Make three lists containing all corresponding CSV file paths that need to be read
    air_csv_list_processed = sorted(glob.glob(os.path.join(AIR_PROCESSED_DATAPATH, "*.csv")))
    paper_csv_list_processed = sorted(glob.glob(os.path.join(PAPER_PROCESSED_DATAPATH, "*.csv")))
    ap_csv_list_processed = sorted(glob.glob(os.path.join(AP_PROCESSED_DATAPATH, "*.csv")))

    # Read all CSV files from the file lists and store them in corresponding dataframe list
    print("Reading files from processed data folder...")
    air_df_list_processed = [ pd.read_csv(csv, sep = SEP) for csv in air_csv_list_processed ]
    paper_df_list_processed = [ pd.read_csv(csv, sep = SEP) for csv in paper_csv_list_processed ]
    ap_df_list_processed = [ pd.read_csv(csv, sep = SEP) for csv in ap_csv_list_processed ]
    
    decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy')
    cross_validator = StratifiedKFold(n_splits = SPLITS, shuffle = True)

    # Run classifier with every dataset
    print("\nRunning classification on air dataset...")
    run_classification_per_ds(decision_tree, cross_validator, air_df_list_processed, 'air')
    print("\nRunning classification on paper dataset...")
    run_classification_per_ds(decision_tree, cross_validator, paper_df_list_processed, 'paper')
    print("\nRunning classification on ap dataset...")
    run_classification_per_ds(decision_tree, cross_validator, air_df_list_processed, 'ap')

    print("")

if __name__ == "__main__":
    main()
