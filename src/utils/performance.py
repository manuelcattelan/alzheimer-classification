def compute_run_results(raw_results):
    for run in raw_results:
        for split_cm, split_runtime in zip(
                raw_results[run][0],
                raw_results[run][1]
                ):
            pass


def compute_clf_results(run_results):
    pass
