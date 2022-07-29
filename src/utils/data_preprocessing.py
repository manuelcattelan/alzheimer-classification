from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from scipy import stats
import numpy as np
import os


def strip_dataframe(df_to_strip):
    df_stripped = df_to_strip.copy()

    # For each feature in features, strip feature
    for feature in ["Sex", "Work", "Label"]:
        df_stripped[feature] = df_to_strip[feature].str.strip()

    return df_stripped


def map_dataframe(df_to_map):
    df_mapped = df_to_map.copy()

    # List of dictionaries that map features from alphabetic to numeric
    sex_map = {"Maschile": 0, "Femminile": 1}
    work_map = {"Manuale": 0, "Intellettuale": 1}
    label_map = {"Sano": 0, "Malato": 1}

    # Perform mapping on all alphabetic features
    df_mapped["Sex"] = df_to_map["Sex"].map(sex_map)
    df_mapped["Work"] = df_to_map["Work"].map(work_map)
    df_mapped["Label"] = df_to_map["Label"].map(label_map)

    return df_mapped


def clean_dataframe(df_to_clean):
    df_cleaned = df_to_clean.copy()

    # Get range of features that need to be checked and cleaned
    features = ["DurationTot", "AveragePenPressureVar"]
    features_range = np.r_[
            df_to_clean.columns.get_loc(features[0]):
            df_to_clean.columns.get_loc(features[1]) + 1
            ]

    # Get list of indexes for all rows that only contain 0's in defined range
    rows_to_clean = df_to_clean.iloc[:, features_range].eq(0).all(1)
    rows_to_clean_index = list(df_to_clean[rows_to_clean].index.values)

    # Drop rows that only contain 0's in defined range
    df_cleaned = df_to_clean.drop(rows_to_clean_index)
    # Drop columnss that only contain 0's
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    # Drop rows that have outliers in at least one column
    df_cleaned = df_cleaned[(np.abs(stats.zscore(df_cleaned)) < 3).all(axis=1)]
    # Drop columns that only contain 0's after removing outliers
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    return df_cleaned


def normalize_dataframe(df_to_normalize):
    df_normalized = df_to_normalize.copy()

    # Get range of features that need to be normalized
    contiguous_features = ["DurationTot", "NumOfStrokes"]
    non_contiguous_features = ["Age", "Instruction"]
    features_range = np.r_[
            df_to_normalize.columns.get_indexer(non_contiguous_features),
            df_to_normalize.columns.get_loc(contiguous_features[0]):
            df_to_normalize.columns.get_loc(contiguous_features[1]) + 1
            ]
    features_names = df_to_normalize.columns[features_range]

    # Initialize and apply MinMaxScaler to dataframe
    scaler = MinMaxScaler()
    df_normalized[features_names] = scaler.fit_transform(
            df_to_normalize[features_names]
            )

    return df_normalized


def preprocess_dataframe(df_raw):
    # List of preprocessing steps:
    # Strip dataframe
    # Map dataframe
    # Clean dataframe
    df_stripped = strip_dataframe(df_raw)
    df_mapped = map_dataframe(df_stripped)
    df_cleaned = clean_dataframe(df_mapped)

    return df_cleaned


def export_dataframe(df_to_export, output_path):
    # Get directory name from output path
    output_dir = Path(os.path.dirname(output_path))
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Export dataframe to output path
    df_to_export.to_csv(output_path, sep=";", index=False)
