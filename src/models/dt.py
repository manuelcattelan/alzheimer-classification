from src.utils.performance import compute_classifier_performance
from src.utils.performance import compute_best_task_performance
from src.utils.classification import run_classification
from src.utils.input import scan_input_dir
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold
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
            "--metric",
            type=str,
            metavar="<p_metric>",
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
                "Classification results for {}:"
                "\n\tClassification time:"
                "\n\t\t>>> Training: {:.5f}s"
                "\n\t\t>>> Testing: {:.5f}s".format(
                    input_path, task_train_time, task_test_time
                    )
                )
        print(
                "\tClassification performance:"
                "\n\t\t>>> Accuracy: {:.3f}%"
                "\n\t\t>>> Precision: {:.3f}%"
                "\n\t\t>>> Recall: {:.3f}%"
                "\n\t\t>>> F1 Score: {:.3f}%".format(
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
                        "Classification results for {}:"
                        "\n\tClassification time:"
                        "\n\t\t>>> Training: {:.5f}s"
                        "\n\t\t>>> Testing: {:.5f}s".format(
                            input_path, task_train_time, task_test_time
                            )
                        )
                print(
                        "\tClassification performance:"
                        "\n\t\t>>> Accuracy: {:.3f}%"
                        "\n\t\t>>> Precision: {:.3f}%"
                        "\n\t\t>>> Recall: {:.3f}%"
                        "\n\t\t>>> F1 Score: {:.3f}%".format(
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

                # for each file inside currently considered dir
                for input_filepath, output_filepath in zip(
                        input_filepaths,
                        output_filepaths
                        ):
                    # read input file as dataframe
                    df = pd.read_csv(input_filepath, sep=";")
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

                # print classification results
                print(
                        "Classification for {} took:"
                        "\n\tTotal classification time (all tasks):"
                        "\n\t\t>>> Training: {:.5f}s"
                        "\n\t\t>>> Testing: {:.5f}s".format(
                            input_dirpath, total_train_time, total_test_time
                            )
                        )
                print(
                        "\tAverage classification time (per task):"
                        "\n\t\t>>> Training: {:.5f}s"
                        "\n\t\t>>> Testing: {:.5f}s".format(
                            avg_train_time, avg_test_time
                            )
                        )
                print(
                        "\tBest performing task for {} was T{}:"
                        "\n\t\tClassification time:"
                        "\n\t\t\t>>> Training: {:.5f}s"
                        "\n\t\t\t>>> Testing: {:.5f}s".format(
                            input_dirpath,
                            best_task_index + 1,
                            best_task_train_time,
                            best_task_test_time
                            )
                        )
                print(
                        "\t\tClassification performance:"
                        "\n\t\t\t>>> Accuracy: {:.3f}%"
                        "\n\t\t\t>>> Precision: {:.3f}%"
                        "\n\t\t\t>>> Recall: {:.3f}%"
                        "\n\t\t\t>>> F1 Score: {:.3f}%".format(
                            best_task_performance[0],
                            best_task_performance[1],
                            best_task_performance[2],
                            best_task_performance[3]
                            )
                        )


if __name__ == '__main__':
    main()
