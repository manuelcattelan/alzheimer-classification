from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def export_classification_results(dt, svm, rf, output):
    # Create output directory for results
    Path(output).mkdir(parents=True, exist_ok=True)

    # Create a dataframe for each performance metric
    accuracy_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    recall_df = pd.DataFrame()

    # For each directory store tasks results in according metric dataframe
    for dt_dir, svm_dir, rf_dir in zip(dt, svm, rf):
        dt_df = pd.DataFrame(dt[dt_dir]).T
        svm_df = pd.DataFrame(svm[svm_dir]).T
        rf_df = pd.DataFrame(rf[rf_dir]).T
        
        # Round all entries in dataframe to 1 decimal
        dt_df = dt_df.round(1)
        svm_df = svm_df.round(1)
        rf_df = rf_df.round(1)

        # Convert all entries in dataframe to string
        dt_df = dt_df.astype(str)
        svm_df = svm_df.astype(str)
        rf_df = rf_df.astype(str)

        dt_dirname = os.path.basename(dt_dir) + "_dt" 
        svm_dirname = os.path.basename(dt_dir) + "_svm" 
        rf_dirname = os.path.basename(rf_dir) + "_rf" 

        # Build dataframes        
        accuracy_df[dt_dirname] = (
                "$" + dt_df["acc_mean"] + "\pm" + dt_df["acc_stdev"] + "$"
                )
        precision_df[dt_dirname] = (
                "$" + dt_df["prec_mean"] + "\pm" + dt_df["prec_stdev"] + "$"
                )
        recall_df[dt_dirname] = (
                "$" + dt_df["rec_mean"] + "\pm" + dt_df["rec_stdev"] + "$"
                )
        accuracy_df[svm_dirname] = (
                "$" + svm_df["acc_mean"] + "\pm" + svm_df["acc_stdev"] + "$"
                )
        precision_df[svm_dirname] = (
                "$" + svm_df["prec_mean"] + "\pm" + svm_df["prec_stdev"] + "$"
                )
        recall_df[svm_dirname] = (
                "$" + svm_df["rec_mean"] + "\pm" + svm_df["rec_stdev"] + "$"
                )
        accuracy_df[rf_dirname] = (
                "$" + rf_df["acc_mean"] + "\pm" + rf_df["acc_stdev"] + "$"
                )
        precision_df[rf_dirname] = (
                "$" + rf_df["prec_mean"] + "\pm" + rf_df["prec_stdev"] + "$"
                )
        recall_df[rf_dirname] = (
                "$" + rf_df["rec_mean"] + "\pm" + rf_df["rec_stdev"] + "$"
                )

    # Build results filepaths
    accuracy_output_path = Path(output) / "accuracy.txt"
    precision_output_path = Path(output) / "precision.txt"
    recall_output_path = Path(output) / "recall.txt"

    # Export results
    accuracy_df.to_latex(buf=accuracy_output_path, escape=False)
    precision_df.to_latex(buf=precision_output_path, escape=False)
    recall_df.to_latex(buf=recall_output_path, escape=False)


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

    # Plot visual configuration
    sns.set_theme(style="ticks", font_scale=0.75)

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
        "Paper": dt_paper_accs,
        "AP": dt_ap_accs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_accs,
        "Paper": svm_paper_accs,
        "AP": svm_ap_accs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_accs,
        "Paper": rf_paper_accs,
        "AP": rf_ap_accs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.tick_params(axis="both", labelsize="small")
    plt.xlabel("Feature type", fontdict={"weight": "bold"})
    plt.ylabel("Accuracy (%)", fontdict={"weight": "bold"})
    plt.title("Accuracy comparison between models")
    plt.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            shadow=True, fancybox=False
            )

    # Export accuracy plot
    plt.savefig(accuracy_plot_path, bbox_inches="tight", dpi=400)
    plt.clf()

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
        "Paper": dt_paper_precs,
        "AP": dt_ap_precs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_precs,
        "Paper": svm_paper_precs,
        "AP": svm_ap_precs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_precs,
        "Paper": rf_paper_precs,
        "AP": rf_ap_precs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.tick_params(axis="both", labelsize="small")
    plt.xlabel("Feature type", fontdict={"weight": "bold"})
    plt.ylabel("Precision (%)", fontdict={"weight": "bold"})
    plt.title("Precision comparison between models")
    plt.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            shadow=True, fancybox=False
            )
    # Export precision plot
    plt.savefig(precision_plot_path, bbox_inches="tight", dpi=400)
    plt.clf()

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
        "Paper": dt_paper_recs,
        "AP": dt_ap_recs,
        "Model": "DT"
        })
    svm_df = pd.DataFrame({
        "Air": svm_air_recs,
        "Paper": svm_paper_recs,
        "AP": svm_ap_recs,
        "Model": "SVM"
        })
    rf_df = pd.DataFrame({
        "Air": rf_air_recs,
        "Paper": rf_paper_recs,
        "AP": rf_ap_recs,
        "Model": "RF"
        })
    # Complete dataframe with all models and all folders
    df = pd.concat([dt_df, svm_df, rf_df])
    # Melt dataframe on "Model" column
    df = pd.melt(df, "Model", var_name="Folder", value_name="Score")
    # Plot configuration
    sns.boxplot(x="Folder", y="Score", hue="Model", data=df, width=0.4)
    plt.tick_params(axis="both", labelsize="small")
    plt.xlabel("Feature type", fontdict={"weight": "bold"})
    plt.ylabel("Recall (%)", fontdict={"weight": "bold"})
    plt.title("Recall comparison between models")
    plt.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            shadow=True, fancybox=False
            )
    # Export recall plot
    plt.savefig(recall_plot_path, bbox_inches="tight", dpi=400)
    plt.close()


def plot_tuning_results(dt, svm, rf, output):
    pass
