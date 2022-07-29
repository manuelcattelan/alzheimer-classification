from sklearn.metrics import confusion_matrix
import time


def train_clf(clf, X, y, train_index):
    # Define training subframes for current split training index:
    # X contains train_index rows with all feature columns
    # y contains train_index rows with Label column
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Train classifier
    start = time.time()
    trained_clf = clf.fit(X_train, y_train)
    stop = time.time()

    train_time = stop - start

    return trained_clf, train_time


def test_clf(clf, X, y, test_index):
    # Define testing subframes for current split training index:
    # X contains test_index rows with all feature columns
    # y contains test_index rows with Label column
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Make predictions on test data
    start = time.time()
    y_pred = clf.predict(X_test)
    stop = time.time()

    test_time = stop - start

    return y_test, y_pred, test_time


def run_clf(clf, cv, df, n_splits, n_runs):
    # Lists that hold each split results
    split_cm_list = []
    split_time_list = []
    # Dictionary that holds split results for each run:
    # [key] = n_run
    # [value] = (split_cm_list, split_time_list)
    clf_results = {}

    # Divide dataframe into two subframes:
    # X contains all feature columns except for the Id and Label column
    # y contains the Label column only
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Initialize loop variables
    split_iter = 1
    run_iter = 1
    # total_split_per_run = n_splits
    # total_split = n_splits * n_runs
    # For each split (total_split):
    for train_index, test_index in cv.split(X, y):
        # Train classifier on train_index
        clf, train_time = train_clf(clf, X, y, train_index)
        # Test classifier on test_index
        true_labels, pred_labels, test_time = test_clf(clf, X, y, test_index)
        # Compute confusion matrix on predictions
        split_cm = confusion_matrix(true_labels, pred_labels)
        # Append results to corresponding lists
        split_cm_list.append(split_cm)
        split_time_list.append((train_time, test_time))
        # If current run is over (total_split_per_run splits were evaluated):
        if split_iter == n_splits:
            # store splits results of current run inside classification results
            clf_results[run_iter] = (split_cm_list, split_time_list)
            # clear splits results
            split_cm_list = []
            split_time_list = []
            # update loop iterators
            split_iter = 1
            run_iter = run_iter + 1
        else:
            split_iter = split_iter + 1

    return clf_results
