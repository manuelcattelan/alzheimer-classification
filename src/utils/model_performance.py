import numpy as np


def compute_runs_report(run_results_dict):
    # Dictionary containing classification results for each run
    runs_report = {}
    # For each performed run
    for run in run_results_dict:
        # Lists holding classification performance metrics
        accuracies = []
        precisions = []
        recalls = []
        # Lists holding classification times
        train_times = []
        test_times = []
        for split_cm, split_times in zip(run_results_dict[run][0],
                                         run_results_dict[run][1]):
            # Ravel currently evaluated confusion matrix
            tn, fp, fn, tp = split_cm.ravel()
            # Compute and append performance metrics to corresponding lists
            accuracies.append((tp + tn) / (tn + fp + fn + tp) * 100)
            if (tp + fp) != 0:
                precisions.append((tp / (tp + fp) * 100))
            else:
                precisions.append(0)
            if (tp + fn) != 0:
                recalls.append((tp / (tp + fn) * 100))
            else:
                recalls.append(0)
            # Append classification times to corresponding lists
            train_times.append(split_times[0])
            test_times.append(split_times[1])
        # Compute mean and variance
        # for all metric arrays
        run_performance = ((np.mean(accuracies), np.var(accuracies)),
                           (np.mean(precisions), np.var(precisions)),
                           (np.mean(recalls), np.var(recalls)))
        # Compute total training and testing times
        run_times = (sum(time for time in train_times),
                     sum(time for time in test_times))
        # Append currently evaluated run results
        # to corresponding entry in classification report
        runs_report[run] = (run_performance, run_times)

    return runs_report


def compute_clf_report(runs_report):
    # List of performance metrics for all runs
    accuracy_mean_list = []
    accuracy_variance_list = []
    precision_mean_list = []
    precision_variance_list = []
    recall_mean_list = []
    recall_variance_list = []
    # List of runtimes for all runs
    train_time_list = []
    test_time_list = []
    # For each executed run in run report
    for run in runs_report:
        # Retrieve run performance metrics from current run report
        acc_mean = runs_report[run][0][0][0]
        acc_var = runs_report[run][0][0][1]
        prec_mean = runs_report[run][0][1][0]
        prec_var = runs_report[run][0][1][1]
        rec_mean = runs_report[run][0][2][0]
        rec_var = runs_report[run][0][2][1]
        # Retrieve runtimes from current run report
        train_time = runs_report[run][1][0]
        test_time = runs_report[run][1][1]

        # Append performance metrics to corresponding lists
        accuracy_mean_list.append(acc_mean)
        precision_mean_list.append(prec_mean)
        recall_mean_list.append(rec_mean)
        accuracy_variance_list.append(acc_var)
        precision_variance_list.append(prec_var)
        recall_variance_list.append(rec_var)
        # Append runtimes to corresponding lists
        train_time_list.append(train_time)
        test_time_list.append(test_time)

    # Compute mean and standard deviation of each performance metric
    accuracy_mean = np.mean(accuracy_mean_list)
    accuracy_stdev = np.sqrt(np.mean(accuracy_variance_list))
    precision_mean = np.mean(precision_mean_list)
    precision_stdev = np.sqrt(np.mean(precision_variance_list))
    recall_mean = np.mean(recall_mean_list)
    recall_stdev = np.sqrt(np.mean(recall_variance_list))
    # Compute total runtimes
    t_train_time = np.sum(train_time_list)
    t_test_time = np.sum(test_time_list)

    return ((accuracy_mean, accuracy_stdev),
            (precision_mean, precision_stdev),
            (recall_mean, recall_stdev),
            (t_train_time, t_test_time))
