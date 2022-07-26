from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from src.utils.scan_input import scan_input_dir
import pandas as pd
import numpy as np
import os


def map_data(df_to_map):
    df_mapped = df_to_map.copy()

    # mapping dictionaries for strings to numeric features
    sex_map = {"Maschile": 0, "Femminile": 1}
    work_map = {"Manuale": 0, "Intellettuale": 1}
    label_map = {"Sano": 0, "Malato": 1}

    # map dataframe with defined dictionaries
    df_mapped["Sex"] = df_to_map["Sex"].map(sex_map)
    df_mapped["Work"] = df_to_map["Work"].map(work_map)
    df_mapped["Label"] = df_to_map["Label"].map(label_map)

    return df_mapped


def clean_data(df_to_clean):
    df_cleaned = df_to_clean.copy()

    # define range of features that need to be cleaned
    # adjacent columns are defined as range between two outmost columns
    adjacent_feature_columns = ["DurationTot", "AveragePenPressureVar"]
    # define numeric features range from outmost feature columns
    features_to_clean = np.r_[
            df_to_clean.columns.get_loc(adjacent_feature_columns[0]):
            df_to_clean.columns.get_loc(adjacent_feature_columns[1]) + 1
            ]
    # create mask for all rows which only contain 0's
    rows_to_drop_mask = (
                df_to_clean.iloc[:, features_to_clean]
                .eq(0)
                .all(1)
                )
    # get list of indexes for rows that match defined mask
    rows_to_drop_index = list(df_to_clean[rows_to_drop_mask].index.values)

    # drop selected rows from dataframe
    df_cleaned = df_to_clean.drop(rows_to_drop_index)
    # drop columns which only contain 0's
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    # drop rows that have outliers in at least one column
    df_cleaned = df_cleaned[(np.abs(stats.zscore(df_cleaned)) < 3).all(axis=1)]
    # drop columns which only contain 0's after removing outliers
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    return df_cleaned


def normalize_data(df_to_normalize):
    df_normalized = df_to_normalize.copy()

    # define range of features that need to be normalized
    # non adjacent columns are the one which are not contiguous
    # adjacent columns are defines as range between two outmost columns
    non_adjacent_feature_columns = ["Age", "Instruction"]
    adjacent_feature_columns = ["DurationTot", "NumOfStrokes"]
    # define numeric features range from feature columns
    features_to_normalize = np.r_[
            df_to_normalize.columns.get_indexer(non_adjacent_feature_columns),
            df_to_normalize.columns.get_loc(adjacent_feature_columns[0]):
            df_to_normalize.columns.get_loc(adjacent_feature_columns[1]) + 1
            ]
    # define features names from range of features
    features_to_normalize_name = df_to_normalize.columns[features_to_normalize]

    # initialize min max scaler and apply it to dataframe to normalize
    scaler = MinMaxScaler()
    df_normalized[features_to_normalize_name] = scaler.fit_transform(
            df_to_normalize[features_to_normalize_name]
            )

    return df_normalized


def export_data(df_to_export, output_path):
    # retrieve dirname of output path and create it if it does not exist
    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    # export df given as argument
    df_to_export.to_csv(output_path, sep=";", index=False)


def run_preprocessing(df_raw):
    # run standard preprocessing
    df_mapped = map_data(df_raw)
    df_cleaned = clean_data(df_mapped)

    return df_cleaned


def file_preprocessing(input_path, output_path):
    # read raw data from input path
    df_to_process = pd.read_csv(
            input_path, sep=";", converters={"Sex": str.strip,
                                             "Work": str.strip,
                                             "Label": str.strip
                                             }
            )
    # preprocess file pointed by input path
    df_processed = run_preprocessing(
            df_to_process
            )
    # export processed file to output path
    export_data(
            df_processed, output_path
            )


def dir_preprocessing(input_path, output_path):
    # get input file path and build corresponding output file path
    # of all files inside input directory
    (input_paths,
     output_paths) = scan_input_dir(
            input_path, output_path
            )
    # for each directory found while traversing input dir
    for input_dirpath, output_dirpath in zip(
            sorted(input_paths),
            sorted(output_paths)):
        # for each file inside currently considered dir
        for input_filepath, output_filepath in zip(
                sorted(input_paths[input_dirpath]),
                sorted(output_paths[output_dirpath])):
            # run single file preprocessing
            file_preprocessing(input_filepath, output_filepath)
