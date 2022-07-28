from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def export_runs_report(input,
                       runs_report,
                       tune_mode,
                       tune_parameters,
                       tune_time,
                       output):
    output_dirname = Path(os.path.dirname(output))
    output_dirname.mkdir(parents=True, exist_ok=True)
    output_with_no_suffix = Path(output).with_suffix("")
    output_with_csv_suffix = Path(output_with_no_suffix).with_suffix(".csv")

    with open(output_with_csv_suffix, 'w+', newline='') as csvfile:
        if tune_mode:
            header = ["n_Run",
                      "Model parameters",
                      "Accuracy mean (%)",
                      "Accuracy stdev (%)",
                      "Precision mean (%)",
                      "Precision stdev (%)",
                      "Recall mean (%)",
                      "Recall stdev (%)",
                      "Train time (s)",
                      "Test time (s)",
                      "Tune time (s)"]
        else:
            header = ["n_Run",
                      "Accuracy mean (%)",
                      "Accuracy stdev (%)",
                      "Precision mean (%)",
                      "Precision stdev (%)",
                      "Recall mean (%)",
                      "Recall stdev (%)",
                      "Train time (s)",
                      "Test time (s)"]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for run in runs_report:
            acc_mean = runs_report[run][0][0][0]
            acc_var = runs_report[run][0][0][1]
            prec_mean = runs_report[run][0][1][0]
            prec_var = runs_report[run][0][1][1]
            rec_mean = runs_report[run][0][2][0]
            rec_var = runs_report[run][0][2][1]
            train_time = runs_report[run][1][0]
            test_time = runs_report[run][1][1]

            if tune_mode:
                data = [run,
                        tune_parameters,
                        format(acc_mean, ".1f"),
                        format(np.sqrt(acc_var), ".1f"),
                        format(prec_mean, ".1f"),
                        format(np.sqrt(prec_var), ".1f"),
                        format(rec_mean, ".1f"),
                        format(np.sqrt(rec_var), ".1f"),
                        format(train_time, ".4f"),
                        format(test_time, ".4f"),
                        format(tune_time, ".4f")]
            else:
                data = [run,
                        format(acc_mean, ".1f"),
                        format(np.sqrt(acc_var), ".1f"),
                        format(prec_mean, ".1f"),
                        format(np.sqrt(prec_var), ".1f"),
                        format(rec_mean, ".1f"),
                        format(np.sqrt(rec_var), ".1f"),
                        format(train_time, ".4f"),
                        format(test_time, ".4f")]

            writer.writerow(data)


def export_clf_report(input, clf_report, output):
    metrics = ["Accuracy", "Precision", "Recall"]

    accuracy_mean = clf_report[0][0]
    accuracy_stdev = clf_report[0][1]
    precision_mean = clf_report[1][0]
    precision_stdev = clf_report[1][1]
    recall_mean = clf_report[2][0]
    recall_stdev = clf_report[2][1]

    CTEs = [accuracy_mean, precision_mean, recall_mean]
    error = [accuracy_stdev, precision_stdev, recall_stdev]

    x_pos = np.arange(len(metrics))
    y_pos = np.arange(0, 100 + 10, 10)

    performance_text = ("\nAccuracy mean={:.1f}%\nAccuracy stdev={:.1f}%"
                        "\n\nPrecision mean={:.1f}%\nPrecision stdev={:.1f}%"
                        "\n\nRecall mean={:.1f}%\nRecall stdev={:.1f}%"
                        .format(accuracy_mean, accuracy_stdev,
                                precision_mean, precision_stdev,
                                recall_mean, recall_stdev))

    fig, ax = plt.subplots()
    ax.bar(x_pos,
           CTEs,
           yerr=error,
           align='center',
           ecolor='black',
           edgecolor='black',
           capsize=5)
    ax.set_title("CLASSIFICATION REPORT FOR '{}'".format(input))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_xlabel(performance_text)
    ax.set_ylabel("Score [%]")
    ax.set_yticks(y_pos)

    output_dirname = Path(os.path.dirname(output))
    output_dirname.mkdir(parents=True, exist_ok=True)
    output_with_no_suffix = Path(output).with_suffix("")
    output_with_png_suffix = Path(output_with_no_suffix).with_suffix(".png")

    plt.tight_layout()
    plt.savefig(output_with_png_suffix, bbox_inches="tight", dpi=400)
    plt.close()
