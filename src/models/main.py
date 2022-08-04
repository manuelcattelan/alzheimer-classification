from src.models.dt import run_dt_classification
from src.models.svm import run_svm_classification
from src.models.rf import run_rf_classification
import argparse
import errno
import os


def main():
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
            help="algorithm to use for hyperparameter tuning",
            choices=["randomized", "grid"]
            )
    parser.add_argument(
            "--iter",
            help="number of iterations for randomized hyperparameter tuning",
            type=int,
            default=60
            )
    parser.add_argument(
            "--jobs",
            help="number of parallel jobs to run during classification",
            type=int,
            default=-1
            )
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

    dt_results = run_dt_classification(args)
    svm_results = run_svm_classification(args)
    rf_results = run_rf_classification(args)

if __name__ == "__main__":
    main()
