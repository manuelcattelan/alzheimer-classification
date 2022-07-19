from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
import numpy as np
import os

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
    # initialize min max scaler and apply to dataframe to normalize
    scaler = MinMaxScaler()
    df_normalized[features_to_normalize] = scaler.fit_transform(df_to_normalize[features_to_normalize])
    df_inverted = df_normalized.copy()
    df_inverted[features_to_normalize] = scaler.inverse_transform(df_normalized[features_to_normalize])

    return df_normalized

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

    return df_normalized
