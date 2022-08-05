from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_classification_results(dt, svm, rf, output):
    # Create output directory for results
    Path(output).mkdir(parents=True, exist_ok=True)
    # Create output paths for different metrics
    accuracy_plot_path = Path(output) / "accuracy.png"
    precision_plot_path = Path(output) / "precision.png"
    recall_plot_path = Path(output) / "recall.png"

    # Folder names
    air = "data/processed/air"
    ap = "data/processed/ap"
    paper = "data/processed/paper"

    # ACCURACY PLOT
    # Accuracies for all folders
    dt_air_accs = [dt[air][task]["acc_mean"] for task in dt[air]]
    svm_air_accs = [svm[air][task]["acc_mean"] for task in svm[air]]
    rf_air_accs = [rf[air][task]["acc_mean"] for task in rf[air]]
    dt_ap_accs = [dt[ap][task]["acc_mean"] for task in dt[ap]]
    svm_ap_accs = [svm[ap][task]["acc_mean"] for task in svm[ap]]
    rf_ap_accs = [rf[ap][task]["acc_mean"] for task in rf[ap]]
    dt_paper_accs = [dt[paper][task]["acc_mean"] for task in dt[paper]]
    svm_paper_accs = [svm[paper][task]["acc_mean"] for task in svm[paper]]
    rf_paper_accs = [rf[paper][task]["acc_mean"] for task in rf[paper]]
    # Model dataframes with results for each folder
    dt_df = pd.DataFrame({
        "Air": dt_air_accs,
        "AP": dt_ap_accs,
        "Paper": dt_paper_accs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_accs,
        "AP": svm_ap_accs,
        "Paper": svm_paper_accs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_accs,
        "AP": rf_ap_accs,
        "Paper": rf_paper_accs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.title("Accuracy comparison between models")
    plt.xlabel("Feature types")
    plt.ylabel("Accuracy (%)")
    # Export accuracy plot
    plt.savefig(accuracy_plot_path, bbox_inches="tight", dpi=400)

    # PRECISION PLOT
    # Precisions for all folders
    dt_air_precs = [dt[air][task]["prec_mean"] for task in dt[air]]
    svm_air_precs = [svm[air][task]["prec_mean"] for task in svm[air]]
    rf_air_precs = [rf[air][task]["prec_mean"] for task in rf[air]]
    dt_ap_precs = [dt[ap][task]["prec_mean"] for task in dt[ap]]
    svm_ap_precs = [svm[ap][task]["prec_mean"] for task in svm[ap]]
    rf_ap_precs = [rf[ap][task]["prec_mean"] for task in rf[ap]]
    dt_paper_precs = [dt[paper][task]["prec_mean"] for task in dt[paper]]
    svm_paper_precs = [svm[paper][task]["prec_mean"] for task in svm[paper]]
    rf_paper_precs = [rf[paper][task]["prec_mean"] for task in rf[paper]]
    # Model dataframes with results for each folder
    dt_df = pd.DataFrame({
        "Air": dt_air_precs,
        "AP": dt_ap_precs,
        "Paper": dt_paper_precs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_precs,
        "AP": svm_ap_precs,
        "Paper": svm_paper_precs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_precs,
        "AP": rf_ap_precs,
        "Paper": rf_paper_precs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.title("Precision comparison between models")
    plt.xlabel("Feature types")
    plt.ylabel("Precision (%)")
    # Export precision plot
    plt.savefig(precision_plot_path, bbox_inches="tight", dpi=400)

    # RECALL PLOT
    # Recalls for all folders
    dt_air_recs = [dt[air][task]["rec_mean"] for task in dt[air]]
    svm_air_recs = [svm[air][task]["rec_mean"] for task in svm[air]]
    rf_air_recs = [rf[air][task]["rec_mean"] for task in rf[air]]
    dt_ap_recs = [dt[ap][task]["rec_mean"] for task in dt[ap]]
    svm_ap_recs = [svm[ap][task]["rec_mean"] for task in svm[ap]]
    rf_ap_recs = [rf[ap][task]["rec_mean"] for task in rf[ap]]
    dt_paper_recs = [dt[paper][task]["rec_mean"] for task in dt[paper]]
    svm_paper_recs = [svm[paper][task]["rec_mean"] for task in svm[paper]]
    rf_paper_recs = [rf[paper][task]["rec_mean"] for task in rf[paper]]
    # Model dataframes with results for each folder
    dt_df = pd.DataFrame({
        "Air": dt_air_recs,
        "AP": dt_ap_recs,
        "Paper": dt_paper_recs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_recs,
        "AP": svm_ap_recs,
        "Paper": svm_paper_recs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_recs,
        "AP": rf_ap_recs,
        "Paper": rf_paper_recs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.title("Recall comparison between models")
    plt.xlabel("Feature types")
    plt.ylabel("Recall (%)")
    # Export recall plot
    plt.savefig(recall_plot_path, bbox_inches="tight", dpi=400)
    

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
