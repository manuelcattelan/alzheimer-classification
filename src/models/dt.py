from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from src.utils.path import build_path
from src.utils.tuning import DT_PARAM_GRID
from src.utils.tuning import tune_clf_params
from src.utils.classification import run_clf
from src.utils.performance import compute_clf_results
from src.utils.performance import compute_best_tasks
from src.utils.performance import compute_worst_tasks
from src.utils.results import plot_tuning_results
from collections import defaultdict
import pandas as pd
import os


def run_dt_classification(args):
    # Check if provided input argument holds path to existing file
    if os.path.isfile(args.input):
        # Read data to classify
        df = pd.read_csv(args.input, sep=";")
        # Initialize classifier and cross validator
        clf = DecisionTreeClassifier(random_state=0)
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats,
                random_state=0
                )
        # If args.tune is defined, tune hyperparameters
        tuning_results = None
        if args.tune is not None:
            clf, tuning_results = tune_clf_params(
                    clf,
                    df,
                    DT_PARAM_GRID,
                    args.jobs
                    )
        # Run classification on data
        raw_results = run_clf(clf, cv, df, args.splits)
        # Compute classification results
        clf_results = compute_clf_results(raw_results)

        return clf_results, tuning_results

    # Check if provided input argument holds path to existing directory
    if os.path.isdir(args.input):
        # Initialize classifier and cross validator
        clf = DecisionTreeClassifier(random_state=0)
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats,
                random_state=0
                )
        # Build input/output paths for each file/directory inside input
        input_paths, output_paths = build_path(
                args.input,
                args.output,
                defaultdict(list),
                defaultdict(list)
                )
        # Dictionary containing results for each dir inside input
        dirs_clf_results = {}
        dirs_tuning_results = {}
        for input_dir, output_dir in zip(
                sorted(input_paths),
                sorted(output_paths)
                ):
            # Dictionary containing results for each file inside current dir
            tasks_clf_results = {}
            tasks_tuning_results = {}
            # For each file inside current dir
            for task_no, (input_file, output_file) in enumerate(
                    zip(
                        sorted(input_paths[input_dir]),
                        sorted(output_paths[output_dir])
                        )
                    ):
                # Read data to classify
                df = pd.read_csv(input_file, sep=";")
                # If args.tune is defined, tune hyperparameters
                tuning_results = None
                if args.tune is not None:
                    clf, tuning_results = tune_clf_params(
                            clf,
                            df,
                            DT_PARAM_GRID,
                            args.jobs
                            )
                # Run classification on data
                raw_results = run_clf(clf, cv, df, args.splits)
                # Compute classification results
                clf_results = compute_clf_results(raw_results)
                # Add classification results to current task's results
                tasks_clf_results[task_no + 1] = clf_results
                # Add tuning results to current task's results
                tasks_tuning_results[task_no + 1] = tuning_results

            dirs_clf_results[input_dir] = tasks_clf_results
            dirs_tuning_results[input_dir] = tasks_tuning_results

        return dirs_clf_results, dirs_tuning_results
