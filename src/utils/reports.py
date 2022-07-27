from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


def print_runs_report(input, runs_report, tune, tune_parameters, tune_time):
    print("CLASSIFICATION REPORT FOR '{}'". format(input))
    if tune:
        print("Tuning took {:.3f}s and produced the following best parameters:"
              .format(tune_time))
        for parameter in tune_parameters:
            print("\t[{}] = {}"
                  .format(parameter, tune_parameters[parameter]))
    else:
        print("TU")
    for run in runs_report:
        acc_mean = runs_report[run][0][0][0]
        acc_var = runs_report[run][0][0][1]
        prec_mean = runs_report[run][0][1][0]
        prec_var = runs_report[run][0][1][1]
        rec_mean = runs_report[run][0][2][0]
        rec_var = runs_report[run][0][2][1]
        train_time = runs_report[run][1][0]
        test_time = runs_report[run][1][1]
        print("Run [{}]:".format(run))
        print("\tAccuracy:"
              "\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(acc_mean, np.sqrt(acc_var)))
        print("\tPrecision:"
              "\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(prec_mean, np.sqrt(prec_var)))
        print("\tRecall:"
              "\t\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(rec_mean, np.sqrt(rec_var)))
        print("\tTimes:"
              "\t\ttraining={:.3f}s"
              "\ttesting={:.3f}s"
              .format(train_time, test_time))


def export_classification_report(input, classification_report, output):
    metrics = ["Accuracy", "Precision", "Recall"]

    accuracy_mean = classification_report[0][0]
    accuracy_stdev = classification_report[0][1]
    precision_mean = classification_report[1][0]
    precision_stdev = classification_report[1][1]
    recall_mean = classification_report[2][0]
    recall_stdev = classification_report[2][1]

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
           alpha=0.5,
           ecolor='black',
           capsize=10)
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
