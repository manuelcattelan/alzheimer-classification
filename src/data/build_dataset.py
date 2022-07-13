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

def build_output_path(input_path, output_path):
    # extract input parent dir and filename from input path
    input_parent_dir = os.path.basename(os.path.dirname(input_path))
    input_filename = os.path.basename(input_path)

    # join initial output path with input parent dir 
    # (e.g. add air/ to data/processed/)
    # and create result directory if it does not exist
    output_dirpath = Path(output_path)
    output_dirpath = output_dirpath / input_parent_dir
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # join new output path with input file name
    # this is the path to which the current file will be saved
    output_filepath = output_dirpath / input_filename

    return output_filepath

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
    args = parser.parse_args()
    args = vars(args)

    # store parsed arguments
    input_path = Path(args['i'])
    output_path = Path(args['o'])

    # check output argument validity by checking
    # if it ends with any extension
    output_path_extension = (os.path.splitext(output_path))[1]

    # check input argument validity by checking
    # if it points to an existing file
    if (os.path.isfile(input_path)):
        if (output_path_extension == '.csv'):
            # build specified input file
            build_data(input_path, output_path)
        else:
            raise ValueError(str(output_path) + ' is not a valid file path' )

    # check input argument validity by checking
    # if it points to an existing directory
    elif (os.path.isdir(input_path)):
        if (output_path_extension == ''):
            # traverse input directory and find all .csv files
            for root, dirs, files in os.walk(input_path):
                if (dirs):
                    for dir in dirs:
                        for file in os.listdir(os.path.join(root, dir)):
                            if (os.path.splitext(file))[1] == '.csv':
                                full_input_path = os.path.join(input_path, dir, file)
                                full_output_path = os.path.join(output_path, dir, file)
                                build_data(full_input_path, full_output_path)
                    break
                elif (files):
                    for file in files:
                        if (os.path.splitext(file))[1] == '.csv':
                            pass
        else:
            raise ValueError(str(output_path) + ' is not a valid directory path')

    # if input argument is neither a file or directory, raise exception
    else:
        raise ValueError(input_path + ' is neither an existing file nor directory')

if __name__ == '__main__':
    main()
