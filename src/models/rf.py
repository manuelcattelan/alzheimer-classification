from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from src.utils.path import build_path
from src.utils.tuning import RF_PARAM_GRID
from src.utils.tuning import RF_PARAM_DISTRIBUTION
from src.utils.tuning import tune_clf_params
from src.utils.classification import run_clf
from src.utils.performance import compute_clf_results
from src.utils.performance import compute_best_task
from src.utils.performance import compute_worst_task
import pandas as pd
import os


def run_rf_classification(args):
    # Check if provided input argument holds path to existing file
    if os.path.isfile(args.input):
        # Read data to classify
        df = pd.read_csv(args.input, sep=";")
        # Initialize classifier and cross validator
        clf = RandomForestClassifier(n_jobs=args.jobs, random_state=0)
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats,
                random_state=0
                )
        # If args.tune is defined, tune hyperparameters
        if args.tune is not None:
            if args.tune == "grid":
                tune_params = RF_PARAM_GRID
            elif args.tune == "randomized":
                tune_params = RF_PARAM_DISTRIBUTION
            clf, tuning_results = tune_clf_params(
                    clf,
                    df,
                    args.tune,
                    tune_params,
                    args.iter,
                    args.jobs
                    )
        # Run classification on data
        raw_results = run_clf(clf, cv, df, args.splits)
        # Compute classification results
        clf_results = compute_clf_results(raw_results)

    # Check if provided input argument holds path to existing directory
    if os.path.isdir(args.input):
        # Initialize classifier and cross validator
        clf = RandomForestClassifier(n_jobs=args.jobs, random_state=0)
        cv = RepeatedStratifiedKFold(
                n_splits=args.splits,
                n_repeats=args.repeats,
                random_state=0
                )
        # Build input/output paths for each file/directory inside input
        input_paths, output_paths = build_path(args.input, args.output)
        # Dictionary containing results for each dir inside input
        dirs_results = {}
        for input_dir, output_dir in zip(
                sorted(input_paths),
                sorted(output_paths)
                ):
            # Dictionary containing results for each file inside current dir
            tasks_results = {}
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
                if args.tune is not None:
                    if args.tune == "grid":
                        tune_params = RF_PARAM_GRID
                    elif args.tune == "randomized":
                        tune_params = RF_PARAM_DISTRIBUTION
                    clf, tuning_results = tune_clf_params(
                            clf,
                            df,
                            args.tune,
                            tune_params,
                            args.iter,
                            args.jobs
                            )
                # Run classification on data
                raw_results = run_clf(clf, cv, df, args.splits)
                # Compute classification results
                clf_results = compute_clf_results(raw_results)
                # Add classification results to current task's results
                tasks_results[task_no + 1] = clf_results

            # Compute best and worst tasks from all tasks results
            best_task = compute_best_task(tasks_results)
            worst_task = compute_worst_task(tasks_results)
            # Add classificaton results to current dir's results
            dirs_results[input_dir] = {
                    "best_task": best_task,
                    "worst_task": worst_task
                    }

        return dirs_results
