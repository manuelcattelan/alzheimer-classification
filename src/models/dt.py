from src.utils.performance import compute_classifier_performance
from src.utils.performance import compute_best_task_performance
from src.utils.classification import run_classification
from src.utils.parameters_tuning import param_distribution
from src.utils.parameters_tuning import param_grid
from src.utils.parameters_tuning import tune_classifier
from src.utils.scan_input import scan_input_dir
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold
import json
import pandas as pd
import numpy as np
import argparse
import os


def main():
    # set up parser and possible arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
    parser.add_argument(
            "--input",
            type=str,
            metavar="<input_file/dir>",
            help=("input path of file to use for classification or directory "
                  "containing files to use for classification"),
            required=True,
            )
    parser.add_argument(
            "--splits",
            type=int,
            metavar="<n_splits>",
            help="number of splits of k-fold cross validation",
            default=10,
            )
    parser.add_argument(
            "--repeats",
            type=int,
            metavar="<n_repeats>",
            help="number of runs of k-fold cross validation",
            default=10,
            )
    parser.add_argument(
            "--tune",
            choices=["randomized", "grid"],
            help="specify algorithm used for tuning hyperparameters",
            )
    parser.add_argument(
            "--metric",
            type=str,
            choices=["accuracy", "precision", "recall", "f1"],
            help=("performance metric used to determine best performing task "
                  "when running dir classification"),
            default="accuracy",
            )
    parser.add_argument(
            "--output",
            type=str,
            metavar="<output_file/dir>",
            help=("output path of file with classification results or "
                  "directory containing files with classification results"),
            required=True,
            )

    # store parsed arguments
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # check output argument validity by checking its extension:
    # if input is file -> output must be file with csv extension
    # if input is dir  -> output must be dir without any extension
    output_path_extension = (os.path.splitext(output_path))[1]

    # if input argument is not an existing file or directory, raise exception
    if (not os.path.isfile(input_path)
            and not os.path.isdir(input_path)):
        raise ValueError(
                input_path + " does not exist as file or directory"
                )

    # if input argument points to file
    if os.path.isfile(input_path):
        # if output argument is not a valid path to png file, raise exception
        if output_path_extension != ".png":
            raise ValueError(
                    output_path + " is not a valid path to .png file"
                    )

        # define classifier and cross validator
        dt = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats
                )
        # read input file as dataframe
        df = pd.read_csv(input_path, sep=";")
        # if tune argument was defined
        if args.tune:
            # based on tune argument option, select corresponding parameters
            params = param_grid if args.tune == "grid" else param_distribution
            # run hyperparameter tuning
            (dt,
             tuning_best_params,
             tuning_time) = tune_classifier(
                    dt, cv, params, df, args.tune
                    )
        # run classification on file
        (splits_cm,
         splits_train_time,
         splits_test_time) = run_classification(
                dt, cv, df
                )
        # compute classifier performance
        (task_performance,
         task_train_time,
         task_test_time) = compute_classifier_performance(
                splits_cm, splits_train_time, splits_test_time
                )
        # print classification results
        print(
                "Classification results for {}:".format(
                    input_path
                    )
                )
        if args.tune:
            # print time taken for hyperparameter tuning
            print(
                    "\tHyperparameters tuning:"
                    "\n\t\t>>> Time taken: {:.3f}s".format(
                        tuning_time
                        )
                    )
        print(
                "\tClassification time:"
                "\n\t\t>>> Training: {:.3f}s"
                "\n\t\t>>> Testing: {:.3f}s".format(
                    task_train_time, task_test_time
                    )
                )
        print(
                "\tClassification performance:"
                "\n\t\t>>> Accuracy: {:.1f}%"
                "\n\t\t>>> Precision: {:.1f}%"
                "\n\t\t>>> Recall: {:.1f}%"
                "\n\t\t>>> F1 Score: {:.1f}%".format(
                    task_performance[0],
                    task_performance[1],
                    task_performance[2],
                    task_performance[3]
                    )
                )

    # check if input argument points to directory
    if os.path.isdir(input_path):
        # if output argument is not a valid path to directory
        if output_path_extension != "":
            raise ValueError(
                    output_path + " is not a valid directory path"
                    )

        # define classifier and cross validator
        dt = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats
                )
        # get input file path and build corresponding output file path
        # of all files inside input directory
        (input_paths,
         output_paths) = scan_input_dir(
                input_path, output_path
                )
        # for each directory found while traversing input dir
        for input_dirpath, output_dirpath in zip(
                sorted(input_paths),
                sorted(output_paths)):

            # list of paths of files inside currently considered dir
            input_filepaths = sorted(input_paths[input_dirpath])
            output_filepaths = sorted(output_paths[output_dirpath])

            # if there's only one file inside the current dir,
            # run single file classification
            if len(input_filepaths) == 1:
                # read input file as dataframe
                df = pd.read_csv(input_filepaths[0], sep=";")
                if args.tune:
                    # based on tune argument option, select corresponding parameters
                    params = param_grid if args.tune == "grid" else param_distribution
                    # run hyperparameter tuning
                    (dt,
                     tuning_best_params,
                     tuning_time) = tune_classifier(
                            dt, cv, params, df, args.tune
                            )
                # run classification on file
                (splits_cm,
                 splits_train_time,
                 splits_test_time) = run_classification(
                        dt, cv, df
                        )
                # compute classifier performance
                (task_performance,
                 task_train_time,
                 task_test_time) = compute_classifier_performance(
                        splits_cm, splits_train_time, splits_test_time
                        )
                # print classification results
                print(
                        "Classification results for {}:".format(
                            input_path
                            )
                        )
                if args.tune:
                    # print time taken for hyperparameter tuning
                    print(
                            "\tHyperparameters tuning:"
                            "\n\t\t>>> Time taken: {:.3f}s".format(
                                tuning_time
                                )
                            )
                print(
                        "\tClassification time:"
                        "\n\t\t>>> Training: {:.3f}s"
                        "\n\t\t>>> Testing: {:.3f}s".format(
                            task_train_time, task_test_time
                            )
                        )
                print(
                        "\tClassification performance:"
                        "\n\t\t>>> Accuracy: {:.1f}%"
                        "\n\t\t>>> Precision: {:.1f}%"
                        "\n\t\t>>> Recall: {:.1f}%"
                        "\n\t\t>>> F1 Score: {:.1f}%".format(
                            task_performance[0],
                            task_performance[1],
                            task_performance[2],
                            task_performance[3]
                            )
                        )

            # if there's at least two files inside current directory,
            # run dir classification
            # this means that also the best classified file will be computed
            # across all files inside dir
            else:
                tasks_performance = []
                tasks_train_time = []
                tasks_test_time = []
                tasks_best_params = []
                tasks_tuning_times = []

                # for each file inside currently considered dir
                for input_filepath, output_filepath in zip(
                        input_filepaths,
                        output_filepaths
                        ):
                    # read input file as dataframe
                    df = pd.read_csv(input_filepath, sep=";")
                    if args.tune:
                        # based on tune argument option, 
                        # select corresponding parameters
                        params = (param_grid if args.tune == "grid" else param_distribution)
                        # run hyperparameter tuning
                        (dt,
                         tuning_best_params,
                         tuning_time) = tune_classifier(
                            dt, cv, params, df, args.tune
                            )
                        tasks_best_params.append(tuning_best_params)
                        tasks_tuning_times.append(tuning_time)
                    # run classification on file
                    (splits_cm,
                     splits_train_time,
                     splits_test_time) = run_classification(
                            dt, cv, df
                            )
                    # compute classifier results
                    (task_performance,
                     task_train_time,
                     task_test_time) = compute_classifier_performance(
                            splits_cm, splits_train_time, splits_test_time
                            )
                    # store classification results
                    tasks_performance.append(task_performance)
                    tasks_train_time.append(task_train_time)
                    tasks_test_time.append(task_test_time)

                # compute best task
                (best_task_performance,
                 best_task_train_time,
                 best_task_test_time,
                 best_task_index) = compute_best_task_performance(
                    tasks_performance,
                    tasks_train_time,
                    tasks_test_time,
                    args.metric
                    )

                # compute classification total time and per avg time per task
                total_train_time = sum(time for time in tasks_train_time)
                total_test_time = sum(time for time in tasks_test_time)
                avg_train_time = np.mean([time for time in tasks_train_time])
                avg_test_time = np.mean([time for time in tasks_test_time])
                # compute classification information regarding tuning
                if args.tune:
                    best_task_params = tasks_best_params[best_task_index]
                    total_tuning_time = sum(time for time in tasks_tuning_times)

                # print classification results
                print(
                        "Classification results for {}:".format(
                                input_dirpath
                                )
                        )
                if args.tune:
                    # print time taken for hyperparameter tuning
                    print(
                            "\tHyperparameters tuning:"
                            "\n\t\t>>> Time taken: {:.3f}s".format(
                                total_tuning_time
                                )
                            )
                print(
                        "\tTotal classification time (all tasks):"
                        "\n\t\t>>> Training: {:.3f}s"
                        "\n\t\t>>> Testing: {:.3f}s".format(
                            total_train_time, total_test_time
                            )
                        )
                print(
                        "\tAverage classification time (per task):"
                        "\n\t\t>>> Training: {:.3f}s"
                        "\n\t\t>>> Testing: {:.3f}s".format(
                            avg_train_time, avg_test_time
                            )
                        )
                print(
                        "\tBest performing task for {} was T{}:"
                        "\n\t\tClassification time:"
                        "\n\t\t\t>>> Training: {:.3f}s"
                        "\n\t\t\t>>> Testing: {:.3f}s".format(
                            input_dirpath,
                            best_task_index + 1,
                            best_task_train_time,
                            best_task_test_time
                            )
                        )
                print(
                        "\t\tClassification performance:"
                        "\n\t\t\t>>> Accuracy: {:.1f}%"
                        "\n\t\t\t>>> Precision: {:.1f}%"
                        "\n\t\t\t>>> Recall: {:.1f}%"
                        "\n\t\t\t>>> F1 score: {:.1f}%".format(
                            best_task_performance[0],
                            best_task_performance[1],
                            best_task_performance[2],
                            best_task_performance[3]
                            )
                        )


if __name__ == '__main__':
    main()
