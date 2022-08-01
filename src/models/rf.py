from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from src.utils.path import build_path
from src.utils.tuning import RF_PARAM_DISTRIBUTION
from src.utils.tuning import tune_clf_params
from src.utils.classification import run_clf
from src.utils.performance import compute_clf_results
from src.utils.results import export_clf_report
from src.utils.results import export_clf_summary
from src.utils.results import export_clf_tuning
import pandas as pd
import argparse
import errno
import os


def main():
    # Setup parser to enable command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument(
            "--input",
            help=("path to file or directory of files where "
                  "data to classify is stored"),
            required=True
            )
    parser.add_argument(
            "--output",
            help=("path to directory where "
                  "classification reports are stored"),
            required=True
            )
    parser.add_argument(
            "--splits",
            type=int,
            help="number of splits (k) for cross validation",
            default=5
            )
    parser.add_argument(
            "--repeats",
            type=int,
            help="number of runs (n) for cross validation",
            default=20
            )
    parser.add_argument(
            "--tune",
            help="tune model hyperparameters using randomized search",
            action=argparse.BooleanOptionalAction
            )
    parser.add_argument(
            "--iter",
            type=int,
            help="number of iterations for randomized hyperparameter tuning",
            default=10
            )
    parser.add_argument(
            "--metric",
            choices=["accuracy", "precision", "recall"],
            help="metric on which hyperparameter tuning is based",
            default=None
            )
    parser.add_argument(
            "--jobs",
            type=int,
            help=("number of jobs to run in parallel during "
                  "classification execution (-1 meaning "
                  "all available processors will be used)"),
            default=None
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

    # Check if provided input argument holds path to existing file
    if os.path.isfile(args.input):
        # read data to classify
        # initialize classifier and cross validator
        # run classification on data
        # eport classification results
        df = pd.read_csv(args.input, sep=";")
        clf = RandomForestClassifier(n_jobs=args.jobs, random_state=0)
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats,
                random_state=0
                )
        # if args.tune is defined, tune hyperparameters
        # before running classification
        if args.tune is not None:
            clf, tuning_results = tune_clf_params(
                    clf,
                    df,
                    RF_PARAM_DISTRIBUTION,
                    args.iter,
                    args.metric,
                    args.jobs
                    )
            export_clf_tuning(tuning_results, args.input, args.output)

        raw_results = run_clf(clf, cv, df, args.splits)
        clf_results = compute_clf_results(raw_results)
        export_clf_report(clf_results, args.input, args.output)
        export_clf_summary(clf_results, args.input, args.output)

    # Check if provided input argument holds path to existing directory
    if os.path.isdir(args.input):
        # Build input/output paths for each file/directory inside args.input
        input_paths, output_paths = build_path(args.input, args.output)
        # For each directory inside args.input
        for input_dir, output_dir in zip(
                sorted(input_paths),
                sorted(output_paths)
                ):
            # For each file inside directory
            for input_file, output_file in zip(
                    sorted(input_paths[input_dir]),
                    sorted(output_paths[output_dir])
                    ):
                # read data to classify
                # initialize classifier and cross validator
                # run classification on data
                # eport classification results
                df = pd.read_csv(input_file, sep=";")
                clf = RandomForestClassifier(n_jobs=args.jobs, random_state=0)
                cv = RepeatedStratifiedKFold(
                        n_splits=args.splits,
                        n_repeats=args.repeats,
                        random_state=0
                        )
                # if args.tune is defined, tune hyperparameters
                # before running classification
                if args.tune is not None:
                    clf, tuning_results = tune_clf_params(
                            clf,
                            df,
                            RF_PARAM_DISTRIBUTION,
                            args.iter,
                            args.metric,
                            args.jobs
                            )
                    export_clf_tuning(tuning_results, input_file, output_dir)

                raw_results = run_clf(clf, cv, df, args.splits)
                clf_results = compute_clf_results(raw_results)
                export_clf_report(clf_results, input_file, output_dir)
                export_clf_summary(clf_results, input_file, output_dir)


if __name__ == "__main__":
    main()
