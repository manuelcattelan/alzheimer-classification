from src.utils.preprocessing import normalize_data
from src.utils.classifier import init_clf
from src.utils.classifier import train_clf
from src.utils.classifier import test_clf
from src.utils.classifier import run_clf
from src.utils.input import input_scan_dict
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold
import argparse
import os

def main():
    # set up parser and possible arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        type=str,
                        metavar='<input_file/dir>',
                        help='input file or directory to classify',
                        required=True)
    parser.add_argument('-s',
                        type=int,
                        metavar='<n_splits>',
                        help='number of splits of k-fold cross validation',
                        default=10)
    parser.add_argument('-r',
                        type=int,
                        metavar='<n_repeats>',
                        help='number of runs of k-fold cross validation',
                        default=10)
    parser.add_argument('-m',
                        type=str,
                        metavar='<p_metric>',
                        help='metric used to determine best performing task when running dir classification',
                        default='accuracy')
    parser.add_argument('-o',
                        type=str,
                        metavar='<output_dir>',
                        help='output directory where results are stored',
                        required=True)

    # store parsed arguments
    args = parser.parse_args()
    input_path = args.i
    output_path = args.o

    # check output argument validity by checking its extension:
    # if input is file -> output must be file with csv extension
    # if input is dir  -> output must be dir without any extension
    output_path_extension = (os.path.splitext(output_path))[1]

    # if input argument is not an existing file or directory, raise exception
    if not(os.path.isfile(input_path)) and not(os.path.isdir(input_path)):
        raise ValueError(str(input_path) + ' is neither an existing file nor directory')

    # check if input argument points to file
    if (os.path.isfile(input_path)):
        # if output argument is not a valid path to png file
        if output_path_extension != '.png':
            raise ValueError(str(output_path) + ' is not a valid png file path')

        # define classifier and cross validator
        clf = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(n_splits=args.s, n_repeats=args.r)

        print('\nRunning {} on {} ...'
                .format(argparse._sys.argv[0], input_path))
        # run classification on file
        results, time = run_clf(clf, cv, input_path, output_path) 
        print('Classification on {} took {:.3f}s:'
              .format(input_path, time))
        print('Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 score: {:.1f}%'
              .format(results[0], results[1], results[2], results[3]))

    # check if input argument points to directory
    if (os.path.isdir(input_path)):
        # if output argument is not a valid path to directory
        if output_path_extension != '':
            raise ValueError(str(output_path) + ' is not a valid directory path')

        # define classifier and cross validator
        clf = tree.DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(n_splits=args.s, n_repeats=args.r)

        print('\nRunning {} on {} ...'
              .format(argparse._sys.argv[0], input_path))
        # recursively scan input directory for any csv file 
        # and store input/output path lists
        input_paths, output_paths = input_scan_dict(input_path, output_path)
        # for each dir inside input argument, make classification on all files inside of it 
        for input_dir, output_dir in zip(input_paths, output_paths):
            # list of files inside dir
            input_filepaths = sorted(input_paths[input_dir])
            output_filepaths = sorted(output_paths[output_dir])

            # if there's only one file inside of dir, run single file classification
            if len(input_filepaths) == 1:
                # run classification on file
                results, time = run_clf(clf, cv, input_filepaths[0], output_filepaths[0])
                print('Classification on {} took {:.3f}s:'
                      .format(input_filepaths[0], time))
                print('Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 score: {:.1f}%'
                      .format(results[0], results[1], results[2], results[3]))

            else:
                # list where each file classification result is stored
                input_results = []
                # list where each file classification time is stored
                input_times = []
                # run classification on each file inside input dir
                for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
                    results, time = run_clf(clf, cv, input_filepath, output_filepath)
                    input_results.append(results)
                    input_times.append(time)

                results, time, index = compute_clf_best_task(input_results, input_times, args['m'])
                total_clf_time = sum([ time for time in input_times ])
                avg_clf_time = np.mean([ time for time in input_times ])

                print('Classification on {} took: {:.3f}s (avg: {:.3f}s)'
                      .format(input_dir, total_clf_time, avg_clf_time))
                print('Best performing task (wrt {}) was T{}, with the following results:'
                      .format(args['m'], index + 1))
                print('Accuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 Score: {:.1f}%\nTime: {:.3f}s'
                      .format(results[0], results[1], results[2], results[3], time))

if __name__ == '__main__':
    main()
