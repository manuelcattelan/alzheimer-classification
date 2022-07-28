from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from src.utils.parameters_tuning import rf_parameters
from src.utils.classification import file_classification
from src.utils.classification import dir_classification
import argparse
import errno
import os


def main():
    # Set up parser to enable possible arguments from command line
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",
                        help=("path to file or directory of files where "
                              "data to classify is stored"),
                        required=True)
    parser.add_argument("--output",
                        help=("path to file or directory of files where "
                              "classification reports are stored"),
                        required=True)
    parser.add_argument("--splits",
                        type=int,
                        help="number of splits (k) for cross validation",
                        default=5)
    parser.add_argument("--runs",
                        type=int,
                        help="number of runs (n) for cross validation",
                        default=20)
    parser.add_argument("--tune",
                        choices=["randomized", "grid"],
                        help="algorithm used to tune model hyperparameters")
    parser.add_argument("--iter",
                        type=int,
                        help="number of iterations for randomized parameter "
                             "tuning (ignored by grid search)",
                        default=10)
    parser.add_argument("--metric",
                        choices=["accuracy", "precision", "recall"],
                        help="metric on which parameter tuning is based",
                        default="accuracy")
    parser.add_argument("--jobs",
                        type=int,
                        help=("number of jobs to run in parallel during "
                              "classification execution (-1 meaning "
                              "all available processors will be used)"),
                        default=1)
    args = parser.parse_args()

    # Check if provided input argument is valid, meaning:
    # Input argument is an existing file, or
    # Input argument is an existing directory
    if not os.path.exists(args.input):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                args.input)

    # Get extension from output argument for later validity checks:
    # If input is a file -> output must end with .png file
    # If input is a directory -> output must end with no extension
    output_arg_extension = os.path.splitext(args.output)[1]

    # Check if provided input argument contains path to file
    if os.path.isfile(args.input):
        if output_arg_extension != ".csv":
            raise ValueError("Not a valid path to csv file: '"
                             + args.output
                             + "'")
        # If everything is OK:
        # run classification on specified input file
        # store classification report on specified output file
        random_forest = RandomForestClassifier(n_jobs=args.jobs)
        cross_validator = RepeatedStratifiedKFold(n_splits=args.splits,
                                                  n_repeats=args.runs)
        file_classification(classifier=random_forest,
                            cross_validator=cross_validator,
                            input_path=args.input,
                            output_path=args.output,
                            normalize=False,
                            jobs=args.jobs,
                            tune_mode=args.tune,
                            tune_iter=args.iter,
                            tune_parameters=rf_parameters,
                            splits=args.splits,
                            metric=args.metric)

    # Check if provided input argument contains path to directory
    if os.path.isdir(args.input):
        if output_arg_extension != "":
            raise ValueError("Not a valid path to directory: '"
                             + args.output
                             + "'")
        # If everything is OK:
        # run classification on specified input directory
        # store classification reports on specified output directory
        random_forest = RandomForestClassifier(n_jobs=args.jobs)
        cross_validator = RepeatedStratifiedKFold(n_splits=args.splits,
                                                  n_repeats=args.runs)
        dir_classification(classifier=random_forest,
                           cross_validator=cross_validator,
                           input_path=args.input,
                           output_path=args.output,
                           normalize=False,
                           jobs=args.jobs,
                           tune_mode=args.tune,
                           tune_iter=args.iter,
                           tune_parameters=rf_parameters,
                           splits=args.splits,
                           metric=args.metric)


if __name__ == '__main__':
    main()
