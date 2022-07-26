import numpy as np


def compute_classification_report(run_results_dict):
    # Dictionary containing classification results for each run
    classification_report = {}
    # For each performed run
    for run in run_results_dict:
        # Lists holding classification performance metrics
        accuracies = []
        precisions = []
        recalls = []
        # Lists holding classification times
        train_times = []
        test_times = []

        for confusion_matrix, times in zip(run_results_dict[run][0],
                                           run_results_dict[run][1]):
            # Ravel currently evaluated confusion matrix
            tn, fp, fn, tp = confusion_matrix.ravel()
            # Compute and append performance metrics to corresponding lists
            accuracies.append((tp + tn) / (tn + fp + fn + tp) * 100)
            precisions.append((tp / (tp + fp) * 100))
            recalls.append((tp / (tp + fn) * 100))
            # Append classification times to corresponding lists
            train_times.append(times[0])
            test_times.append(times[1])

        # Compute mean and standard deviation
        # for all metric arrays
        run_performance = ((np.mean(accuracies), np.std(accuracies)),
                           (np.mean(precisions), np.std(precisions)),
                           (np.mean(recalls), np.std(recalls)))
        # Compute total training and testing times
        run_times = (sum(time for time in train_times),
                     sum(time for time in test_times))
        # Append currently evaluated run results
        # to corresponding entry in classification report
        classification_report[run] = (run_performance,
                                      run_times)

    return classification_report
