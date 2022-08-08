from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import numpy as np
import os


def export_classification_results(dt, svm, rf, output):
    # Create output directory for results
    Path(output).mkdir(parents=True, exist_ok=True)

    for dir in dt:
        df = pd.DataFrame(dt[dir]).T
        df = df.round(1)
        df = df.astype(str)
        df["accuracy (\%)"] = "$" + df["acc_mean"] + "\pm" + df["acc_stdev"] + "$"
        df["precision (\%)"] = "$" + df["prec_mean"] + "\pm" + df["prec_stdev"] + "$"
        df["recall (\%)"] = "$" + df["rec_mean"] + "\pm" + df["rec_stdev"] + "$"
        df = df.drop(
                ["acc_mean", "acc_stdev",
                 "prec_mean", "prec_stdev",
                 "rec_mean", "rec_stdev",
                 "train_time", "test_time"],
                axis=1
                )
        dt_results_path = output + "/"  + os.path.basename(dir) + "_dt.txt"
        df.to_latex(buf=dt_results_path, escape=False)

    for dir in svm:
        df = pd.DataFrame(svm[dir]).T
        df = df.round(1)
        df = df.astype(str)
        df["accuracy (\%)"] = "$" + df["acc_mean"] + "\pm" + df["acc_stdev"] + "$"
        df["precision (\%)"] = "$" + df["prec_mean"] + "\pm" + df["prec_stdev"] + "$"
        df["recall (\%)"] = "$" + df["rec_mean"] + "\pm" + df["rec_stdev"] + "$"
        df = df.drop(
                ["acc_mean", "acc_stdev",
                 "prec_mean", "prec_stdev",
                 "rec_mean", "rec_stdev",
                 "train_time", "test_time"],
                axis=1
                )
        svm_results_path = output + "/"  + os.path.basename(dir) + "_svm.txt"
        df.to_latex(buf=svm_results_path, escape=False)
        

    for dir in rf:
        df = pd.DataFrame(rf[dir]).T
        df = df.round(1)
        df = df.astype(str)
        df["accuracy (\%)"] = "$" + df["acc_mean"] + "\pm" + df["acc_stdev"] + "$"
        df["precision (\%)"] = "$" + df["prec_mean"] + "\pm" + df["prec_stdev"] + "$"
        df["recall (\%)"] = "$" + df["rec_mean"] + "\pm" + df["rec_stdev"] + "$"
        df = df.drop(
                ["acc_mean", "acc_stdev",
                 "prec_mean", "prec_stdev",
                 "rec_mean", "rec_stdev",
                 "train_time", "test_time"],
                axis=1
                )
        rf_results_path = output + "/"  + os.path.basename(dir) + "_rf.txt"
        df.to_latex(buf=rf_results_path, escape=False)


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
    total_df = pd.DataFrame()
    for dir in svm:
        for task in svm[dir]:
            task_df = pd.DataFrame(svm[dir][task])[
                    [
                        "params",
                        "mean_test_score",
                        "std_test_score",
                        "mean_fit_time",
                        "mean_score_time",
                        "rank_test_score",
                        ]
                    ]
            total_df = pd.concat([total_df, task_df])

    # Get 10% of rows number in dataframe
    rows_to_plot = round(total_df.shape[0] * 10/100)
    # Expand params column into multiple individual columns
    total_df = (
            pd.DataFrame(total_df.pop("params").values.tolist())
            ).join(total_df)
    # Get top 10% rows in dataframe
    total_df = total_df.nlargest(
            rows_to_plot, "mean_test_score"
            ).sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # # Build dataframe containing tuning results
    # tuning_res = pd.DataFrame(tuning_res)[
    #         [
    #             "params",
    #             "mean_test_score",
    #             "std_test_score",
    #             "mean_fit_time",
    #             "mean_score_time",
    #             "rank_test_score",
    #             ]
    #         ]
    # # Get 10% of rows number in dataframe
    # rows_to_plot = round(tuning_res.shape[0] * 10/100)
    # # Expand params column into multiple individual columns
    # tuning_res = (
    #         pd.DataFrame(tuning_res.pop("params").values.tolist())
    #         ).join(tuning_res)
    # # Get top 10% rows in dataframe
    # tuning_res = tuning_res.nsmallest(
    #         rows_to_plot, "rank_test_score", keep="first"
    #         )
    # Define columns that are present for every model (all except params)
    default_cols = [
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "mean_score_time",
            "rank_test_score",
            ]
    # Define columns that contain parameters
    param_cols = [col for col in df.keys() if not col in default_cols]
    # Define columns that contain categorical data
    categorical_cols = df[param_cols].select_dtypes(
            include=["object", "bool"]
            ).columns
    col_list = []
    for col in param_cols:
        if col in categorical_cols:
            values = df[col].unique()
            dummy_values = dict(zip(values, range(len(values))))
            df[col] = [dummy_values[value] for value in df[col]]
            col_dict = dict(
                    label=col.capitalize().replace("_", " "),
                    tickvals=list(dummy_values.values()),
                    ticktext=list(dummy_values.keys()),
                    values=df[col],
                    )
        else:
            col_dict = dict(
                    label=col.capitalize().replace("_", " "),
                    values=df[col],
                    )
        col_list.append(col_dict)

    col_list.append(dict(
        label="Score",
        values=df["mean_test_score"],
        ))

    line = dict(
            color=df["mean_test_score"].astype("float"),
            showscale=True,
            )
    fig = go.Figure(
            data=go.Parcoords(
                line=line,
                dimensions=col_list,
                )
            )

    fig.update_layout(width=1200, height=800)
    fig.show()
