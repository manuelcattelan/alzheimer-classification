from src.utils.path import build_path
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from pathlib import Path
import pandas as pd
import numpy as np
import os


def map_data(df_to_map):
    df_mapped = df_to_map.copy()

    # Mapping dictionaries for alphabetical features
    sex_map = {"Maschile": 0, "Femminile": 1}
    work_map = {"Manuale": 0, "Intellettuale": 1}
    label_map = {"Sano": 0, "Malato": 1}

    # Map alphabetical features to numeric features
    df_mapped["Sex"] = df_to_map["Sex"].map(sex_map)
    df_mapped["Work"] = df_to_map["Work"].map(work_map)
    df_mapped["Label"] = df_to_map["Label"].map(label_map)

    return df_mapped


def clean_data(df_to_clean):
    df_cleaned = df_to_clean.copy()

    # Define range of features that need to be cleaned:
    # adjacent feature columns indicate features that are contiguous
    adjacent_feature_columns = ["DurationTot", "AveragePenPressureVar"]
    # Define numeric range for features that need to be cleaned
    features_to_clean = np.r_[
            df_to_clean.columns.get_loc(adjacent_feature_columns[0]):
            df_to_clean.columns.get_loc(adjacent_feature_columns[1]) + 1]
    # Create boolean mask for all rows that
    # only contain 0 in the defined feature range
    rows_to_clean_mask = df_to_clean.iloc[:, features_to_clean].eq(0).all(1)
    # Get a list of indexes for rows that match mask
    rows_to_clean_index = list(df_to_clean[rows_to_clean_mask].index.values)

    # Drop rows that match mask from dataframe
    df_cleaned = df_to_clean.drop(rows_to_clean_index)
    # Drop columns that only contain 0's
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    # Drop rows that have outliers in at least one column
    df_cleaned = df_cleaned[(np.abs(stats.zscore(df_cleaned)) < 3).all(axis=1)]
    # Drop columns which only contain 0's after removing outliers
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    return df_cleaned


def normalize_data(df_to_normalize):
    df_normalized = df_to_normalize.copy()

    # Define range of features that need to be normalized:
    # non adjacent feature columns indicate features that are not contiguous
    # adjacent feature columns indicate features that are contiguous
    non_adjacent_feature_columns = ["Age", "Instruction"]
    adjacent_feature_columns = ["DurationTot", "NumOfStrokes"]
    # Define numeric range for features that need to be normalized
    features_to_normalize = np.r_[
            df_to_normalize.columns.get_indexer(non_adjacent_feature_columns),
            df_to_normalize.columns.get_loc(adjacent_feature_columns[0]):
            df_to_normalize.columns.get_loc(adjacent_feature_columns[1]) + 1]
    # Define names for features that need to be normalized
    features_to_normalize_name = df_to_normalize.columns[features_to_normalize]

    # Initialize and apply MinMaxScaler to dataframe
    scaler = MinMaxScaler()
    df_normalized[features_to_normalize_name] = scaler.fit_transform(
            df_to_normalize[features_to_normalize_name])

    return df_normalized


def export_data(df_to_export, output_path):
    # Get directory name from output path
    output_dirname = Path(os.path.dirname(output_path))
    # Create output directory if it does not exist
    output_dirname.mkdir(parents=True, exist_ok=True)
    # Export dataframe to output directory
    df_to_export.to_csv(output_path, sep=";", index=False)


def run_preprocessing(df_raw):
    # Run data preprocessing steps
    df_mapped = map_data(df_to_map=df_raw)
    df_cleaned = clean_data(df_to_clean=df_mapped)

    return df_cleaned


def file_preprocessing(input_path, output_path):
    # Read raw data from input path into dataframe
    df_to_preprocess = pd.read_csv(input_path,
                                   sep=";",
                                   converters={"Sex": str.strip,
                                               "Work": str.strip,
                                               "Label": str.strip})
    # Preprocess raw dataframe
    df_preprocessed = run_preprocessing(df_raw=df_to_preprocess)
    # Export preprocessed dataframe to output path
    export_data(df_to_export=df_preprocessed,
                output_path=output_path)


def dir_preprocessing(input_path, output_path):
    # Recursively scan input path in order to:
    # build a list of all input paths to read
    # build a list of all corresponding output paths to write
    input_path_list, output_path_list = build_path(input_path, output_path)
    # For each directory path found while traversing input path
    for input_dirpath, output_dirpath in zip(sorted(input_path_list),
                                             sorted(output_path_list)):
        # For each file path found inside currently considered directory
        for input_filepath, output_filepath in zip(
                sorted(input_path_list[input_dirpath]),
                sorted(output_path_list[output_dirpath])):
            # Run preprocessing on specified input file
            # Store preprocessed data on specified output file
            file_preprocessing(input_path=input_filepath,
                               output_path=output_filepath)
