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


def compute_best_tasks(results):
    # Flatten results dictionary
    results_list = list(results.values())

    # Compute tasks with best metrics and find corresponding task numbers
    max_acc_task = max(results_list, key=lambda x: x['acc_mean'])
    max_acc_task_no = list(results.keys())[results_list.index(max_acc_task)]
    max_prec_task = max(results_list, key=lambda x: x['prec_mean'])
    max_prec_task_no = list(results.keys())[results_list.index(max_prec_task)]
    max_rec_task = max(results_list, key=lambda x: x['rec_mean'])
    max_rec_task_no = list(results.keys())[results_list.index(max_rec_task)]

    return (
            (max_acc_task_no, max_acc_task['acc_mean']),
            (max_prec_task_no, max_prec_task['prec_mean']),
            (max_rec_task_no, max_rec_task['rec_mean'])
            )


def compute_worst_tasks(results):
    # Flatten results dictionary
    results_list = list(results.values())

    # Compute tasks with best metrics and find corresponding task numbers
    min_acc_task = min(results_list, key=lambda x: x['acc_mean'])
    min_acc_task_no = list(results.keys())[results_list.index(min_acc_task)]
    min_prec_task = min(results_list, key=lambda x: x['prec_mean'])
    min_prec_task_no = list(results.keys())[results_list.index(min_prec_task)]
    min_rec_task = min(results_list, key=lambda x: x['rec_mean'])
    min_rec_task_no = list(results.keys())[results_list.index(min_rec_task)]

    return (
            (min_acc_task_no, min_acc_task['acc_mean']),
            (min_prec_task_no, min_prec_task['prec_mean']),
            (min_rec_task_no, min_rec_task['rec_mean'])
            )
