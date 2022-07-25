from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from src.utils.parameters_tuning import dt_params
from src.utils.classification import file_classification
from src.utils.classification import dir_classification
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
        dt = DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats
                )
        # run classification process on input file
        file_classification(
                dt,
                cv,
                input_path,
                output_path,
                dt_params,
                args.tune
                )

    # check if input argument points to directory
    if os.path.isdir(input_path):
        # if output argument is not a valid path to directory
        if output_path_extension != "":
            raise ValueError(
                    output_path + " is not a valid directory path"
                    )
        # define classifier and cross validator
        dt = DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats
                )
        # run classification process on input directory
        dir_classification(
                dt,
                cv,
                input_path,
                output_path,
                dt_params,
                args.tune,
                args.metric
                )


if __name__ == '__main__':
    main()
