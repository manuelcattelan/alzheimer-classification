import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_results(dt_results, svm_results, rf_results=None):
    dt_best_accuracies = []
    dt_worst_accuracies = []
    svm_best_accuracies = []
    svm_worst_accuracies = []
    for dt_dir, svm_dir in zip(dt_results, svm_results):
       dt_best_accuracies.append(
               dt_results[dt_dir]["best_tasks"][0][1]["acc_mean"]
               ) 
       dt_worst_accuracies.append(
               dt_results[dt_dir]["worst_tasks"][0][1]["acc_mean"]
               )
       svm_best_accuracies.append(
               svm_results[svm_dir]["best_tasks"][0][1]["acc_mean"]
               ) 
       svm_worst_accuracies.append(
               svm_results[svm_dir]["worst_tasks"][0][1]["acc_mean"]
               )

    labels = ["AIR", "PAPER", "AP"]
    df = pd.DataFrame(
            {
                "DT": dt_best_accuracies,
                "SVM": svm_best_accuracies
                },
            index=labels
        ) 
    ax = df.plot(kind="bar")
    plt.show()
    # labels = ["AIR", "PAPER", "AP"]
    # x = np.arange(len(labels))
    # width = 0.35

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, dt_best_accuracies, width, label="DT")
    # rects2 = ax.bar(x + width/2, svm_best_accuracies, width, label="SVM")
    # ax.set_xticks(x, labels)
    # ax.legend()

    # fig.tight_layout()
    # plt.show()
