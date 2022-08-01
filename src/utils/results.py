from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def export_clf_report(clf_results, output_path):
    # Get directory name from output path
    output_dir = Path(os.path.dirname(output_path))
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Add csv extension to output path
    output_with_no_suffix = Path(output_path).with_suffix("")
    output_with_no_suffix = str(output_with_no_suffix) + "_report"
    output_with_csv_suffix = Path(output_with_no_suffix).with_suffix(".csv")

    # Open output file in write mode or create it if it does not exist
    with open(output_with_csv_suffix, "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Define header row
        header = [
                "Run",
                "Accuracy [%]",
                "Precision [%]",
                "Recall [%]",
                "Train time [s]",
                "Test time [s]",
                ]
        # Write header row
        writer.writerow(header)

        # For each run result in clf_results
        for run_no in clf_results:
            # Define data row
            data = [
                    run_no,
                    "{:.2f} (±{:.2f})".format(
                        clf_results[run_no][0][0] * 100,
                        clf_results[run_no][1][0] * 100
                        ),
                    "{:.2f} (±{:.2f})".format(
                        clf_results[run_no][0][1] * 100,
                        clf_results[run_no][1][1] * 100
                        ),
                    "{:.2f} (±{:.2f})".format(
                        clf_results[run_no][0][2] * 100,
                        clf_results[run_no][1][2] * 100
                        ),
                    "{:.4f}".format(clf_results[run_no][2][0]),
                    "{:.4f}".format(clf_results[run_no][2][1]),
                    ]
            # Write data row
            writer.writerow(data)


def export_clf_summary(clf_results, output_path):
    # Compute averaged results for classification
    runs_accuracy_mean = sum(
            means[0][0]
            for means in clf_results.values()
            ) / float(len(clf_results)) * 100
    runs_accuracy_stdev = np.sqrt(
            sum(stds[1][0] ** 2
                for stds in clf_results.values()
                ) / float(len(clf_results))
            ) * 100
    runs_precision_mean = sum(
            means[0][1]
            for means in clf_results.values()
            ) / float(len(clf_results)) * 100
    runs_precision_stdev = np.sqrt(
            sum(stds[1][1] ** 2
                for stds in clf_results.values()
                ) / float(len(clf_results))
            ) * 100
    runs_recall_mean = sum(
            means[0][2]
            for means in clf_results.values()
            ) / float(len(clf_results)) * 100
    runs_recall_stdev = np.sqrt(
            sum(stds[1][2] ** 2
                for stds in clf_results.values()
                ) / float(len(clf_results))
            ) * 100
    total_train_time = sum(
            time[2][0] for time in clf_results.values()
            )
    total_test_time = sum(
            time[2][1] for time in clf_results.values()
            )

    # Define list of performance metrics to put in x-axis
    metrics = ["Accuracy", "Precision", "Recall"]

    # Build list of metrics mean and standard deviation
    CTEs = [runs_accuracy_mean, runs_precision_mean, runs_recall_mean]
    error = [runs_accuracy_stdev, runs_precision_stdev, runs_recall_stdev]

    x_pos = np.arange(len(metrics))
    y_pos = np.arange(0, 100 + 10, 10)

    # Build annotation text to show precise performance metrics values
    performance_text = ("\nAccuracy [%]: {:.2f} (±{:.2f})\n"
                        "Precision [%]: {:.2f} (±{:.2f})\n"
                        "Recall [%]: {:.2f} (±{:.2f})\n\n"
                        "Train time [s]: {:.4f}\n"
                        "Test time [s]: {:.4f}".format(
                                runs_accuracy_mean, runs_accuracy_stdev,
                                runs_precision_mean, runs_precision_stdev,
                                runs_recall_mean, runs_recall_stdev,
                                total_train_time, total_test_time
                            )
                        )
    # Build bar plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error,
           align='center',
           ecolor='black',
           edgecolor='black',
           capsize=5)
    # Configure bar plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_xlabel(performance_text)
    ax.set_ylabel("Score [%]")
    ax.set_yticks(y_pos)

    # Extract dirname from output path and
    # create directory if it does not exist
    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    # Add png extension to output path
    output_with_no_suffix = Path(output_path).with_suffix("")
    output_with_no_suffix = str(output_with_no_suffix) + "_summary"
    output_with_png_suffix = Path(output_with_no_suffix).with_suffix(".png")

    # Export and close plot
    plt.tight_layout()
    plt.savefig(output_with_png_suffix, bbox_inches="tight", dpi=400)
    plt.close()


def export_clf_tuning(tuning_results, output_path):
    # Extract dirname from output path and
    # create directory if it does not exist
    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    # Add csv extension to output path
    output_with_no_suffix = Path(output_path).with_suffix("")
    output_with_no_suffix = str(output_with_no_suffix) + "_tuning"
    output_with_csv_suffix = Path(output_with_no_suffix).with_suffix(".csv")

    tuning_results.to_csv(output_with_csv_suffix)