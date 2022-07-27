import matplotlib.pyplot as plt
import numpy as np


def print_runs_report(input, runs_report):
    print("RUNS REPORT FOR '{}'". format(input))
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

def plot_classification_report(input, classification_report):
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

    fig, ax = plt.subplots()
    ax.bar(x_pos,
           CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize = 10)
    ax.set_ylabel("Classification result")
    ax.set_xticks(x_pos)
    ax.set_yticks(y_pos)
    ax.set_xticklabels(metrics)
    ax.set_title("Classification report for " + input)

    plt.tight_layout()
    plt.show()
