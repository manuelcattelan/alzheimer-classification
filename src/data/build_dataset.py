from pathlib import Path
from scipy import stats
import pandas as pd
import numpy as np
import argparse
import glob
import os

def clean_data(df_to_clean):
    df_to_clean = df_to_clean.copy()

    # get range of features to clean
    contiguous_columns = ['DurationTot', 'AveragePenPressureVar']
    features_to_clean_range = np.r_[df_to_clean.columns.get_loc(contiguous_columns[0]):
                                    df_to_clean.columns.get_loc(contiguous_columns[1]) + 1]
    # create mask for all rows which only contain 0's
    rows_to_drop_mask = df_to_clean.iloc[:, features_to_clean_range].eq(0).all(1)
    # get list of indexes for rows that match mask
    rows_to_drop = list(df_to_clean[rows_to_drop_mask].index.values)

    # drop selected rows from raw dataframe
    df_cleaned = df_to_clean.drop(rows_to_drop)
    # # drop columns which only contain 0's
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    # remove rows that have outliers in at least one column
    df_cleaned = df_cleaned[(np.abs(stats.zscore(df_cleaned)) < 3).all(axis=1)]
    # # drop columns which only contain 0's
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    return df_cleaned

def normalize_data(df_to_normalize):
    df_to_normalize = df_to_normalize.copy()
    df_normalized = df_to_normalize.copy()

    # get range of features that need to be normalized
    non_contiguous_columns = ['Age', 'Instruction']
    contiguous_columns = ['DurationTot', 'NumOfStrokes']
    features_to_normalize_range = np.r_[df_to_normalize.columns.get_indexer(non_contiguous_columns),
                                        df_to_normalize.columns.get_loc(contiguous_columns[0]):
                                        df_to_normalize.columns.get_loc(contiguous_columns[1]) + 1] 
    # get features names from range of features
    features_to_normalize = df_to_normalize.columns[features_to_normalize_range]

    # iterate over each feature that needs to be normalized
    for feature in features_to_normalize:
        # get min and max values from feature column
        max_value = df_to_normalize[feature].max()
        min_value = df_to_normalize[feature].min()
        # if feature column only contains 0s, store 0.0
        if (max_value - min_value) == 0:
            df_normalized[feature] = float(0)
        # if feature column does not contain 0s only, replace each entry with normalized value
        else:
            df_normalized[feature] = (df_normalized[feature] - min_value) / (max_value - min_value)

    return df_normalized

def map_data(df_to_map):
    df_to_map = df_to_map.copy() 
    df_mapped = df_to_map.copy()

    # mapping dictionaries for string to numeric features
    sex_map = {'Maschile': 0, 'Femminile': 1}
    work_map = {'Manuale': 0, 'Intellettuale': 1}
    label_map = {'Sano': 0, 'Malato': 1}

    # map dataframe with mapping dictionaries
    df_mapped['Sex'] = df_to_map['Sex'].map(sex_map)
    df_mapped['Work'] = df_to_map['Work'].map(work_map)
    df_mapped['Label'] = df_to_map['Label'].map(label_map)

    return df_mapped

def build_data(input_path, output_path):
    # read input path as dataframe
    df_raw = pd.read_csv(input_path, sep=';', converters={'Sex': str.strip,
                                                          'Work': str.strip,
                                                          'Label': str.strip})
    # entire preprocessing
    df_mapped = map_data(df_raw)
    df_cleaned = clean_data(df_mapped)
    df_normalized = normalize_data(df_cleaned)

    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    df_normalized.to_csv(output_path, sep=';', index=False)

def recursive_input_scan(input_root, output_root, input_paths=[], output_paths=[]):
    # scan input root dir (find files and subdirs)
    input_root_content = os.listdir(input_root)

    for content in input_root_content:
        # for each content inside input root dir, build content absolute paths
        input_content_path = os.path.join(input_root, content)
        output_content_path = os.path.join(output_root, content)
        # if content is file in csv format, append absolute paths in path lists 
        if (os.path.isfile(input_content_path) and
            os.path.splitext(input_content_path)[1] == '.csv'):
            input_paths.append(input_content_path)
            output_paths.append(output_content_path)
        # if content is dir, recursively call this function to scan dir content
        elif (os.path.isdir(input_content_path) and
              not(input_content_path.startswith('.'))):
            # update root input path and root output path for recursive call
            new_input_root = input_content_path
            new_output_root = output_content_path
            recursive_input_scan(new_input_root, new_output_root, input_paths, output_paths)

    return sorted(input_paths), sorted(output_paths)

def main():
    # set up parser and possible arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i',
                        type=str,
                        metavar='<input_file/dir>',
                        help='input file or directory to process',
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        metavar='<output_file/dir>',
                        help='output file or directory where processed data is saved',
                        required=True)

    # store parsed arguments
    args = parser.parse_args()
    args = vars(args)
    input_path = Path(args['i'])
    output_path = Path(args['o'])

    # check output argument validity by checking its extension:
    # if input is file -> output must be file with csv extension
    # if input is dir  -> output must be dir without any extension
    output_path_extension = (os.path.splitext(output_path))[1]

    # check input argument validity by checking
    # if it points to an existing file
    if (os.path.isfile(input_path)):
        # if output is a valid path to file
        if (output_path_extension == '.csv'):
            # build specified input file
            build_data(input_path, output_path)
        else:
            raise ValueError(str(output_path) + ' is not a valid file path' )

    # check input argument validity by checking
    # if it points to an existing directory
    elif (os.path.isdir(input_path)):
        # if output is a valid path to folder
        if (output_path_extension == ''):
            # recursively scan input folder for any csv file 
            # and store input/output path lists
            input_paths, output_paths = recursive_input_scan(input_path, output_path)
            # for each input/output path pair, build input and export to output
            for input_path, output_path in zip(input_paths, output_paths):
                build_data(input_path, output_path)
        else:
            raise ValueError(str(output_path) + ' is not a valid directory path')

    # if input argument is neither a file or directory, raise exception
    else:
        raise ValueError(input_path + ' is neither an existing file nor directory')

if __name__ == '__main__':
    main()
