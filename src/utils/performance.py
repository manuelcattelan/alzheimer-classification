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

    # Open output file in write mode or create it if it does not exist
    with open(output_with_csv_suffix, "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Define header row
        header = [
                "run_no",
                "parameters",
                "splits_accuracy_mean (%)",
                "splits_accuracy_stdev (%)",
                "splits_precision_mean (%)",
                "splits_precision_stdev (%)",
                "splits_recall_mean (%)",
                "splits_recall_stdev (%)",
                "splits_train_time (s)",
                "splits_test_time (s)",
                "splits_tune_time (s)"
                ]
        # Write header row
        writer.writerow(header)

        # For each run result in clf_results
        for run_no in clf_results:
            # Define data row
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
            # Write data row
            writer.writerow(data)

        # Compute average classification results from row results
        runs_accuracy_mean = sum(
                means[0][0]
                for means in clf_results.values()
                ) / float(len(clf_results))
        runs_accuracy_stdev = np.sqrt(
                sum(vars[1][0]
                    for vars in clf_results.values()
                    ) / float(len(clf_results))
                )
        runs_precision_mean = sum(
                means[0][1]
                for means in clf_results.values()
                ) / float(len(clf_results))
        runs_precision_stdev = np.sqrt(
                sum(vars[1][1]
                    for vars in clf_results.values()
                    ) / float(len(clf_results))
                )
        runs_recall_mean = sum(
                means[0][2]
                for means in clf_results.values()
                ) / float(len(clf_results))
        runs_recall_stdev = np.sqrt(
                sum(vars[1][2]
                    for vars in clf_results.values()
                    ) / float(len(clf_results))
                )
        total_train_time = sum(
                time[2][0] for time in clf_results.values()
                )
        total_test_time = sum(
                time[2][1] for time in clf_results.values()
                )
        # Define summary row
        summary = [
                "SUMMARY",
                clf_best_params if tune_time is not None else "default",
                runs_accuracy_mean,
                runs_accuracy_stdev,
                runs_precision_mean,
                runs_precision_stdev,
                runs_recall_mean,
                runs_recall_stdev,
                total_train_time,
                total_test_time,
                clf_best_params if tune_time is not None else 0,
                ]
        # Write summary row
        writer.writerow(summary)
