def compute_classifier_performance(
        split_cm,
        split_train_time,
        split_test_time
        ):
    # compute task classifier performance from split results
    task_cm = sum(cm for cm in split_cm)
    task_train_time = sum(time for time in split_train_time)
    task_test_time = sum(time for time in split_test_time)

    # get each matrix cell separately
    tn, fp, fn, tp = task_cm.ravel()

    # compute task matrix performance
    accuracy = ((tp + tn) / (tn + fp + fn + tp) * 100)
    precision = (tp / (tp + fp) * 100)
    recall = (tp / (tp + fn) * 100)
    f1_score = (2 * precision * recall / (precision + recall))

    return (
            (accuracy, precision, recall, f1_score),
            task_train_time,
            task_test_time
            )


def compute_best_task_performance(
        tasks_performance,
        tasks_train_time,
        tasks_test_time,
        p_metric
        ):
    # map metric argument to corresponging index in performance tuple
    metric = {'accuracy': 0, 'precision': 1, 'recall': 2, 'f1': 3}[p_metric]

    # Compute best task classification information
    best_task_metric_value = max(
            [
                results[metric]
                for results
                in tasks_performance
                ]
            )
    best_task_index = (
            [
                results[metric]
                for results
                in tasks_performance
                ]
            .index(best_task_metric_value)
            )
    best_task_train_time = tasks_train_time[best_task_index]
    best_task_test_time = tasks_test_time[best_task_index]

    # Obtain best task performances
    best_task_accuracy = tasks_performance[best_task_index][0]
    best_task_precision = tasks_performance[best_task_index][1]
    best_task_recall = tasks_performance[best_task_index][2]
    best_task_f1score = tasks_performance[best_task_index][3]

    return (
            (best_task_accuracy,
             best_task_precision,
             best_task_recall,
             best_task_f1score
             ),
            best_task_train_time,
            best_task_test_time,
            best_task_index
            )
