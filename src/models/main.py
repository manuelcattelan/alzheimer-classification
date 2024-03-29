from src.models.dt import run_dt_classification
from src.models.svm import run_svm_classification
from src.models.rf import run_rf_classification
from src.utils.results import export_classification_results
from src.utils.results import plot_classification_results
from src.utils.results import plot_tuning_results
import argparse
import errno
import os


def main():
    # Set up parser and add possible arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter
            )
    parser.add_argument(
            "--input",
            help="file or directory of files containing data to classify",
            required=True
            )
    parser.add_argument(
            "--output",
            help="directory where classification results are exported",
            required=True
            )
    parser.add_argument(
            "--splits",
            help="number of splits (k) for cross validation",
            type=int,
            default=5
            )
    parser.add_argument(
            "--repeats",
            help="number of repeats (n) for cross validation",
            type=int,
            default=20
            )
    parser.add_argument(
            "--tune",
            help="tune hyperparameters using GridSearch",
            action=argparse.BooleanOptionalAction
            )
    parser.add_argument(
            "--jobs",
            help="number of parallel jobs to run during classification",
            type=int,
            default=-1
            )
    # Parse arguments
    args = parser.parse_args()

    # Check if provided input argument is valid, meaning:
    # args.input is a path to an existing file, or
    # args.input is a path to an existing directory
    if not os.path.exists(args.input):
        raise FileExistsError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                args.input
                )

    # Check if provided output argument is valid:
    # it has to hold a valid path to directory in order
    # to store classification results inside of it
    output_extension = os.path.splitext(args.output)[1]
    if output_extension != "":
        raise ValueError(
                "not a valid path to directory: '"
                + args.output
                + "'"
                )

    dt_clf_results, dt_tuning_results = run_dt_classification(args)
    svm_clf_results, svm_tuning_results = run_svm_classification(args)
    rf_clf_results, rf_tuning_results = run_rf_classification(args)
    export_classification_results(
            dt_clf_results, svm_clf_results, rf_clf_results,
            args.output
            )
    plot_classification_results(
            dt_clf_results, svm_clf_results, rf_clf_results,
            args.output
            )
    if args.tune is not None:
        plot_tuning_results(
                dt_tuning_results, svm_tuning_results, rf_tuning_results,
                args.output
                )


if __name__ == "__main__":
    main()
