import numpy as np


def compute_clf_results(raw_results):
    run_results = {}
    # For each run
    for run_no in raw_results:
        # Compute mean over all splits for each performance metric
        run_performance_mean = [
                sum(metric)/len(raw_results[run_no][0])
                for metric in zip(*raw_results[run_no][0])
                ]
        # Compute variance over all splits for each performance metric
        run_performance_std = [
                np.std(metric)
                for metric in zip(*raw_results[run_no][0])
                ]
        # Compute total runtime from all splits
        run_runtime = [
                sum(time) for time in zip(*raw_results[run_no][1])
                ]
        # Add results to [run_no] in dictionary
        run_results[run_no] = (
                run_performance_mean,
                run_performance_std,
                run_runtime
                )

    # Compute averaged results from all runs
    accuracy_mean = sum(
            means[0][0]
            for means in run_results.values()
            ) / float(len(run_results)) * 100
    accuracy_stdev = np.sqrt(
            sum(stds[1][0] ** 2
                for stds in run_results.values()
                ) / float(len(run_results))
            ) * 100
    precision_mean = sum(
            means[0][1]
            for means in run_results.values()
            ) / float(len(run_results)) * 100
    precision_stdev = np.sqrt(
            sum(stds[1][1] ** 2
                for stds in run_results.values()
                ) / float(len(run_results))
            ) * 100
    recall_mean = sum(
            means[0][2]
            for means in run_results.values()
            ) / float(len(run_results)) * 100
    recall_stdev = np.sqrt(
            sum(stds[1][2] ** 2
                for stds in run_results.values()
                ) / float(len(run_results))
            ) * 100
    train_time = sum(
            time[2][0] for time in run_results.values()
            )
    test_time = sum(
            time[2][1] for time in run_results.values()
            )

    # Create results dictionary
    clf_results = {
            "acc_mean": accuracy_mean,
            "acc_stdev": accuracy_stdev,
            "prec_mean": precision_mean,
            "prec_stdev": precision_stdev,
            "rec_mean": recall_mean,
            "rec_stdev": recall_stdev,
            "train_time": train_time,
            "test_time": test_time
            }

    return clf_results


def compute_best_tasks(tasks_results):
    # Flatten results dictionary
    tasks_results_values = tasks_results.values()
    # Compute tasks with best metrics
    best_accuracy = max(
            tasks_results_values,
            key=lambda results : results['acc_mean']
            )
    best_accuracy_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(best_accuracy)
            ]
    best_precision = max(
            tasks_results_values,
            key=lambda results : results['prec_mean']
            )
    best_precision_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(best_precision)
            ]
    best_recall = max(
            tasks_results_values,
            key=lambda results : results['rec_mean']
            )
    best_recall_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(best_recall)
            ]

    return (
            (best_accuracy_task, best_accuracy),
            (best_precision_task, best_precision),
            (best_recall_task, best_recall)
            )


def compute_worst_tasks(tasks_results):
    # Flatten results dictionary
    tasks_results_values = tasks_results.values()
    # Compute tasks with worst metrics
    worst_accuracy = min(
            tasks_results_values,
            key=lambda results : results['acc_mean']
            )
    worst_accuracy_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(worst_accuracy)
            ]
    worst_precision = min(
            tasks_results_values,
            key=lambda results : results['prec_mean']
            )
    worst_precision_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(worst_precision)
            ]
    worst_recall = min(
            tasks_results_values,
            key=lambda results : results['rec_mean']
            )
    worst_recall_task = list(tasks_results.keys())[
            list(tasks_results.values()).index(worst_recall)
            ]

    return (
            (worst_accuracy_task, worst_accuracy),
            (worst_precision_task, worst_precision),
            (worst_recall_task, worst_recall)
            )
