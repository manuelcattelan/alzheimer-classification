def compute_clf_performance(cm):
    # get each matrix cell separately
    tn, fp, fn, tp = cm.ravel()

    # compute task performance
    accuracy = ((tp + tn) / (tn + fp + fn + tp) * 100)
    precision = (tp / (tp + fp) * 100)
    recall = (tp / (tp + fn) * 100)
    f1_score = (2 * precision * recall / (precision + recall))

    return (accuracy, precision, recall, f1_score)

def compute_clf_best_task(tasks_results, tasks_times, p_metric):
    # map metric argument to corresponging index in performance tuple
    metric = {'accuracy': 0, 'precision': 1, 'recall': 2, 'f1': 3}[p_metric]

    # Compute best task classification information
    best_task_metric = max([ results[metric] for results in tasks_results ])
    best_task_index = [ results[metric] for results in tasks_results ].index(best_task_metric)
    best_task_time = tasks_times[best_task_index]

    # Obtain best task performances
    best_task_accuracy = tasks_results[best_task_index][0] 
    best_task_precision = tasks_results[best_task_index][1] 
    best_task_recall = tasks_results[best_task_index][2] 
    best_task_f1score = tasks_results[best_task_index][3]

    return (best_task_accuracy, best_task_precision, best_task_recall, best_task_f1score), best_task_time, best_task_index
