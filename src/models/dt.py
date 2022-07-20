from src.utils.performance import compute_classifier_performance
from src.utils.performance import compute_best_task_performance
from src.utils.classifier import run_classification
from src.utils.input import scan_input_to_dict
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold
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
            help="input path to file or directory to which classification is performed",
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
            help="metric used to determine best performing task when running dir classification",
            default="accuracy",
    )
    parser.add_argument(
            "--output",
            type=str,
            metavar="<output_file/dir>",
            help="output path to file or directory to which results are stored",
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
    if not(os.path.isfile(input_path)) and not(os.path.isdir(input_path)):
        raise ValueError(str(input_path) + " is neither an existing file nor directory")

    # check if input argument points to file
    if (os.path.isfile(input_path)):
        # if output argument is not a valid path to png file
        if output_path_extension != ".png":
            raise ValueError(str(output_path) + " is not a valid png file path")

        # define classifier and cross validator
        dt = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats)

        # run classification on file
        splits_cm_list, splits_train_time_list, splits_test_time_list = run_classification(dt, cv, input_path)
        task_performance, task_train_time, task_test_time = compute_classifier_performance(splits_cm_list, splits_train_time_list, splits_test_time_list)
        print("Classification results for {}:\n\tTimes:\n\t\t>>> Training time: {:.5f}s\n\t\t>>> Testing time: {:.5f}s".format(input_path, task_train_time, task_test_time))
        print("\tPerformance:\n\t\t>>> Accuracy: {:.3f}%\n\t\t>>> Precision: {:.3f}%\n\t\t>>> Recall: {:.3f}%\n\t\t>>> F1 Score: {:.3f}%".format(
            task_performance[0], task_performance[1], task_performance[2], task_performance[3]))

    # check if input argument points to directory
    if (os.path.isdir(input_path)):
        # if output argument is not a valid path to directory
        if output_path_extension != "":
            raise ValueError(str(output_path) + " is not a valid directory path")

        # define classifier and cross validator
        dt = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats)

        # recursively scan input directory for any csv file 
        # and store input/output path lists
        input_paths, output_paths = scan_input_to_dict(input_path, output_path)
        # for each dir inside input argument, make classification on all files inside of it 
        for input_dir, output_dir in zip(input_paths, output_paths):
            # list of files inside dir
            input_filepaths = sorted(input_paths[input_dir])
            output_filepaths = sorted(output_paths[output_dir])

            # if there's only one file inside of dir, run single file classification
            if len(input_filepaths) == 1:
                # run classification on file
                task_cm_list, task_train_time_list, task_test_time_list = run_classification(dt, cv, input_path)
                task_performance, task_train_time, task_test_time = compute_classifier_performance(splits_cm_list, splits_train_time_list, splits_test_time_list)
                print("Classification results for {}:\n\tTimes:\n\t\t>>> Training time: {:.5f}s\n\t\t>>> Testing time: {:.5f}s".format(input_path, task_train_time, task_test_time))
                print("\tPerformance:\n\t\t>>> Accuracy: {:.3f}%\n\t\t>>> Precision: {:.3f}%\n\t\t>>> Recall: {:.3f}%\n\t\t>>> F1 Score: {:.3f}%".format(
                    task_performance[0], task_performance[1], task_performance[2], task_performance[3]))

            else:
                tasks_performance = []
                tasks_train_time = []
                tasks_test_time = []
                # run classification on each file inside input dir
                for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
                    splits_cm_list, splits_train_time_list, splits_test_time_list = run_classification(dt, cv, input_filepath)
                    task_performance, task_train_time, task_test_time = compute_classifier_performance(splits_cm_list, splits_train_time_list, splits_test_time_list)
                    tasks_performance.append(task_performance)
                    tasks_train_time.append(task_train_time)
                    tasks_test_time.append(task_test_time)

                best_task_performance, best_task_train_time, best_task_test_time, best_task_index = compute_best_task_performance(tasks_performance, tasks_train_time, tasks_test_time, args.metric)

                total_training_time = sum(time for time in tasks_train_time)
                total_testing_time = sum(time for time in tasks_test_time)
                avg_training_time = np.mean([ time for time in tasks_train_time])
                avg_testing_time = np.mean([ time for time in tasks_test_time])
                
                print("Classification for {} took:\n\tTotal times (all tasks)\n\t\t>>> {:.5f}s for training\n\t\t>>> {:.5f}s for testing".format(input_dir, total_training_time, total_testing_time))
                print("\tAverage times (per task)\n\t\t>>> {:.5f}s for training\n\t\t>>> {:.5f}s for testing".format(avg_training_time, avg_testing_time))
                print("Best performing task for {} was T{}:\n\tTimes:\n\t\t>>> Training time: {:.5f}s\n\t\t>>> Testing time: {:.5f}s".format(input_dir, best_task_index, best_task_train_time, best_task_test_time))
                print("\tPerformance:\n\t\t>>> Accuracy: {:.3f}%\n\t\t>>> Precision: {:.3f}%\n\t\t>>> Recall: {:.3f}%\n\t\t>>> F1 Score: {:.3f}%".format(
                    best_task_performance[0], best_task_performance[1], best_task_performance[2], best_task_performance[3]))


if __name__ == '__main__':
    main()
