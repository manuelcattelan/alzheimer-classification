import numpy as np


def compute_clf_results(raw_results):
    clf_results = {}
    # For each run in raw_results
    for run_no in raw_results:
        # Compute mean over all splits for each performance metric
        run_performance_mean = [
                sum(metric)/len(raw_results[run_no][0]) 
                for metric in zip(*raw_results[run_no][0])
                ]
        # Compute variance over all splits for each performance metric
        run_performance_var = [
                np.var(metric)
                for metric in zip(*raw_results[run_no][0])
                ]
        # Compute total runtime from all splits
        run_runtime = [
                sum(time) for time in zip(*raw_results[run_no][1])
                ]
        # Add results to [run_no] in dictionary
        clf_results[run_no] = (
                (run_performance_mean, run_performance_var),
                run_runtime
                )
    return clf_results
