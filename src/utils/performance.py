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
            "recall_mean": recall_mean,
            "recall_stdev": recall_stdev,
            "train_time": train_time,
            "test_time": test_time
            }

    return clf_results


def compute_best_task(tasks_results):
    # Flatten results dictionary
    tasks_results_values = tasks_results.values()
    # Compute task with maximum mean accuracy
    max_results = max(
            tasks_results_values,
            key=lambda results : results['acc_mean']
            )
    # Retrieve best task index from tasks results
    best_task_no = list(tasks_results.keys())[
            list(tasks_results.values()).index(max_results)
            ]

    return (best_task_no, max_results)


def compute_worst_task(tasks_results):
    # Flatten results dictionary
    tasks_results_values = tasks_results.values()
    # Compute task with minimum mean accuracy
    min_results = min(
            tasks_results_values,
            key=lambda results : results['acc_mean']
            )
    # Retrieve worst task index from tasks results
    worst_task_no = list(tasks_results.keys())[
            list(tasks_results.values()).index(min_results)
            ]

    return (worst_task_no, min_results)
