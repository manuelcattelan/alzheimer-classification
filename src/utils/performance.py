from pathlib import Path
import numpy as np
import csv
import os


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
                run_performance_mean,
                run_performance_var,
                run_runtime
                )
    return clf_results


def export_clf_results(clf_results, clf_best_params, tune_time, output_path):
    # Get directory name from output path
    output_dir = Path(os.path.dirname(output_path))
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Add csv extension to output path
    output_with_no_suffix = Path(output_path).with_suffix("")
    output_with_csv_suffix = Path(output_with_no_suffix).with_suffix(".csv")

    with open(output_with_csv_suffix, "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [
                "run_no",
                "parameters",
                "acc_mean[%]",
                "acc_stdev[%]",
                "prec_mean[%]",
                "prec_stdev[%]",
                "rec_mean[%]",
                "rec_stdev[%]",
                "train_time[s]",
                "test_time[s]",
                "tune_time[s]"
                ]
        writer.writerow(header)
        for run_no in clf_results:
            data = [
                    run_no,
                    clf_best_params if tune_time is not None else "default",
                    "{:%}".format(clf_results[run_no][0][0]),
                    "{:%}".format(np.sqrt(clf_results[run_no][1][0])),
                    "{:%}".format(clf_results[run_no][0][1]),
                    "{:%}".format(np.sqrt(clf_results[run_no][1][1])),
                    "{:%}".format(clf_results[run_no][0][2]),
                    "{:%}".format(np.sqrt(clf_results[run_no][1][2])),
                    "{:.4f}".format(clf_results[run_no][2][0]),
                    "{:.4f}".format(clf_results[run_no][2][1]),
                    "{:.4f}".format(tune_time) if tune_time is not None else 0,
                    ]
            writer.writerow(data)
