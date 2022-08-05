from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_classification_results(dt_res, svm_res, rf_res, output):
    # Create output directory for results
    Path(output).mkdir(parents=True, exist_ok=True)
    # Create output paths for different metrics
    accuracy_plot_path = Path(output) / "accuracy.png"
    precision_plot_path = Path(output) / "precision.png"
    recall_plot_path = Path(output) / "recall.png"

    # Get lists of best tasks for all folders and for all metrics
    dt_best_acc = [dt_res[key]["best_tasks"][0][1] for key in dt_res]
    dt_best_prec = [dt_res[key]["best_tasks"][1][1] for key in dt_res]
    dt_best_rec = [dt_res[key]["best_tasks"][2][1] for key in dt_res]
    svm_best_acc = [svm_res[key]["best_tasks"][0][1] for key in svm_res]
    svm_best_rec = [svm_res[key]["best_tasks"][2][1] for key in svm_res]
    svm_best_prec = [svm_res[key]["best_tasks"][1][1] for key in svm_res]
    # rf_best_acc = [rf_res[key]["best_tasks"][0][1] for key in rf_res]
    # rf_best_rec = [rf_res[key]["best_tasks"][2][1] for key in rf_res]
    # rf_best_prec = [rf_res[key]["best_tasks"][1][1] for key in rf_res]

    # Get lists of worst tasks for all folders and for all metrics
    dt_worst_acc = [dt_res[key]["worst_tasks"][0][1] for key in dt_res]
    dt_worst_prec = [dt_res[key]["worst_tasks"][1][1] for key in dt_res]
    dt_worst_rec = [dt_res[key]["worst_tasks"][2][1] for key in dt_res]
    svm_worst_acc = [svm_res[key]["worst_tasks"][0][1] for key in svm_res]
    svm_worst_rec = [svm_res[key]["worst_tasks"][2][1] for key in svm_res]
    svm_worst_prec = [svm_res[key]["worst_tasks"][1][1] for key in svm_res]
    # rf_best_acc = [rf_res[key]["best_tasks"][0][1] for key in rf_res]
    # rf_best_rec = [rf_res[key]["best_tasks"][2][1] for key in rf_res]
    # rf_best_prec = [rf_res[key]["best_tasks"][1][1] for key in rf_res]

    # Positions for every group
    labels = ["Air", "Paper", "Ap"]
    width = 0.25
    x1 = np.arange(len(labels))
    x2 = [x + width for x in x1]
    x3 = [x + width for x in x2]

    # ACCURACY PLOT
    # Plot every bar in the corresponding group
    fig, ax = plt.subplots()
    dt_best_accs = ax.bar(x1, dt_best_acc, width, edgecolor="black")
    dt_worst_accs = ax.bar(x1, dt_worst_acc, width, edgecolor="black")
    svm_best_accs = ax.bar(x2, svm_best_acc, width, edgecolor="black")
    svm_worst_accs = ax.bar(x2, svm_worst_acc, width, edgecolor="black")
    rf_best_accs = ax.bar(x3, [80, 90, 70], width, edgecolor="black")
    rf_worst_accs = ax.bar(x3, [65, 70, 60], width, edgecolor="black")

    # Configure ticks
    ticks_pos = [rect.xy[0] + rect.get_width()/2. for rect in ax.patches]
    ticks_pos = np.unique(ticks_pos)
    ax.set_xticks(
            ticks_pos,
            ["DT", "SVM", "RF", "DT", "SVM", "RF", "DT", "SVM", "RF"],
            minor=True
            )
    ax.set_xticks([r + width for r in range(len(labels))], labels)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Scores (%)")
    ax.set_title("Accuracy comparison between classifiers")
    ax.tick_params(axis="x", which="minor", labelsize="x-small")
    ax.tick_params(axis="x", which="major", labelsize="medium", pad=15)
    ax.xaxis.remove_overlapping_locs = False

    # Configure legends
    dt_legend = ax.legend(
            (dt_best_accs, dt_worst_accs), ("Best task", "Worst task"),
            title="DT", loc="upper left",
            bbox_to_anchor=(1.01, 1),
            )
    ax.add_artist(dt_legend)
    svm_legend = ax.legend(
            (svm_best_accs, svm_worst_accs), ("Best task", "Worst task"),
            title="SVM", loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            )
    ax.add_artist(svm_legend)
    rf_legend = ax.legend(
            (rf_best_accs, rf_worst_accs), ("Best task", "Worst task"),
            title="RF", loc="lower left",
            bbox_to_anchor=(1.01, 0),
            )
    ax.add_artist(rf_legend)

    # Configure annotations
    # ax.bar_label(dt_best_accs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(dt_worst_accs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_best_accs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_worst_accs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_best_accs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_worst_accs, fmt="%3.1f", fontweight="medium", padding=2)

    fig.tight_layout()

    # Export accuracy plot
    plt.savefig(accuracy_plot_path, bbox_inches="tight", dpi=400)

    # # PRECISION PLOT
    # # Plot every bar in the corresponding group
    # fig, ax = plt.subplots()
    # dt_best_precs = ax.bar(x1, dt_best_prec, width, edgecolor="black")
    # dt_worst_precs = ax.bar(x1, dt_worst_prec, width, edgecolor="black")
    # svm_best_precs = ax.bar(x2, svm_best_prec, width, edgecolor="black")
    # svm_worst_precs = ax.bar(x2, svm_worst_prec, width, edgecolor="black")
    # rf_best_precs = ax.bar(x3, [80, 90, 70], width, edgecolor="black")
    # rf_worst_precs = ax.bar(x3, [65, 70, 60], width, edgecolor="black")

    # # Configure ticks
    # ticks_pos = [rect.xy[0] + rect.get_width()/2. for rect in ax.patches]
    # ticks_pos = np.unique(ticks_pos)
    # ax.set_xticks(
    #         ticks_pos,
    #         ["DT", "SVM", "RF", "DT", "SVM", "RF", "DT", "SVM", "RF"],
    #         minor=True
    #         )
    # ax.set_xticks([r + width for r in range(len(labels))], labels)
    # ax.set_ylim([0, 100])
    # ax.set_ylabel("Scores (%)")
    # ax.set_title("Precision comparison between classifiers")
    # ax.tick_params(axis="x", which="minor", labelsize="x-small")
    # ax.tick_params(axis="x", which="major", labelsize="medium", pad=15)
    # ax.xaxis.remove_overlapping_locs = False

    # # Configure legends
    # dt_legend = ax.legend(
    #         (dt_best_precs, dt_worst_precs), ("Best task", "Worst task"),
    #         title="DT", loc="upper left",
    #         bbox_to_anchor=(1.01, 1),
    #         )
    # ax.add_artist(dt_legend)
    # svm_legend = ax.legend(
    #         (svm_best_precs, svm_worst_precs), ("Best task", "Worst task"),
    #         title="SVM", loc="center left",
    #         bbox_to_anchor=(1.01, 0.5),
    #         )
    # ax.add_artist(svm_legend)
    # rf_legend = ax.legend(
    #         (rf_best_precs, rf_worst_precs), ("Best task", "Worst task"),
    #         title="RF", loc="lower left",
    #         bbox_to_anchor=(1.01, 0),
    #         )
    # ax.add_artist(rf_legend)

    # # Configure annotations
    # ax.bar_label(dt_best_precs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(dt_worst_precs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_best_precs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_worst_precs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_best_precs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_worst_precs, fmt="%3.1f", fontweight="medium", padding=2)

    # fig.tight_layout()

    # # Export precision plot
    # plt.savefig(precision_plot_path, bbox_inches="tight", dpi=400)

    # # RECALL PLOT
    # # Plot every bar in the corresponding group
    # fig, ax = plt.subplots()
    # dt_best_recs = ax.bar(x1, dt_best_rec, width, edgecolor="black")
    # dt_worst_recs = ax.bar(x1, dt_worst_rec, width, edgecolor="black")
    # svm_best_recs = ax.bar(x2, svm_best_rec, width, edgecolor="black")
    # svm_worst_recs = ax.bar(x2, svm_worst_rec, width, edgecolor="black")
    # rf_best_recs = ax.bar(x3, [80, 90, 70], width, edgecolor="black")
    # rf_worst_recs = ax.bar(x3, [65, 70, 60], width, edgecolor="black")

    # # Configure ticks
    # ticks_pos = [rect.xy[0] + rect.get_width()/2. for rect in ax.patches]
    # ticks_pos = np.unique(ticks_pos)
    # ax.set_xticks(
    #         ticks_pos,
    #         ["DT", "SVM", "RF", "DT", "SVM", "RF", "DT", "SVM", "RF"],
    #         minor=True
    #         )
    # ax.set_xticks([r + width for r in range(len(labels))], labels)
    # ax.set_ylim([0, 100])
    # ax.set_ylabel("Scores (%)")
    # ax.set_title("Recall comparison between classifiers")
    # ax.tick_params(axis="x", which="minor", labelsize="x-small")
    # ax.tick_params(axis="x", which="major", labelsize="medium", pad=15)
    # ax.xaxis.remove_overlapping_locs = False

    # # Configure legends
    # dt_legend = ax.legend(
    #         (dt_best_recs, dt_worst_recs), ("Best task", "Worst task"),
    #         title="DT", loc="upper left",
    #         bbox_to_anchor=(1.01, 1),
    #         )
    # ax.add_artist(dt_legend)
    # svm_legend = ax.legend(
    #         (svm_best_recs, svm_worst_recs), ("Best task", "Worst task"),
    #         title="SVM", loc="center left",
    #         bbox_to_anchor=(1.01, 0.5),
    #         )
    # ax.add_artist(svm_legend)
    # rf_legend = ax.legend(
    #         (rf_best_recs, rf_worst_recs), ("Best task", "Worst task"),
    #         title="RF", loc="lower left",
    #         bbox_to_anchor=(1.01, 0),
    #         )
    # ax.add_artist(rf_legend)

    # # Configure annotations
    # ax.bar_label(dt_best_recs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(dt_worst_recs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_best_recs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(svm_worst_recs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_best_recs, fmt="%3.1f", fontweight="medium", padding=2)
    # ax.bar_label(rf_worst_recs, fmt="%3.1f", fontweight="medium", padding=2)

    # fig.tight_layout()

    # # Export recall plot
    # plt.savefig(recall_plot_path, bbox_inches="tight", dpi=400)


def plot_tuning_results(tuning_res, output):
    # Build dataframe containing tuning results
    tuning_res = pd.DataFrame(tuning_res)[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "mean_score_time",
                "rank_test_score",
                ]
            ]
    # Expand params column into multiple individual columns
    tuning_res = (
            pd.DataFrame(tuning_res.pop("params").values.tolist())
            ).join(tuning_res)
    # Sort results by tuning rank
    tuning_res = tuning_res.sort_values("rank_test_score")
    print(tuning_res)
