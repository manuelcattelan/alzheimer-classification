# Modules
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import glob
import os

# Folder paths for raw dataset 
AIR_RAW_DATAPATH = "../../data/raw/air/"
PAPER_RAW_DATAPATH = "../../data/raw/paper/"
AP_RAW_DATAPATH = "../../data/raw/ap/"
# Folder path for processed dataset
AIR_PROCESSED_DATAPATH = "../../data/processed/air/"
PAPER_PROCESSED_DATAPATH = "../../data/processed/paper/"
AP_PROCESSED_DATAPATH = "../../data/processed/ap/"

# Column range holding features that need to be normalized (numeric features)
TO_NORMALIZE_FEATURES_RANGE = np.r_[ 1:87, 88, 90 ]
# Column range holding features that need to be filtered out
TO_CLEAN_FEATURES_RANGE = np.r_[ 1:86 ]
# Separator character used to read input data
SEP = "," 

# Drop erroneous data acquisition rows by checking rows with all 0 values
def clean_data(df_to_clean):
    # Create copy of dataframe given as parameter
    df_to_clean = df_to_clean.copy()

    # Create mask for all rows with only value 0 in TO_CLEAN_FEATURES_RANGE range
    rows_to_drop_mask = df_to_clean.iloc[:, TO_CLEAN_FEATURES_RANGE].eq(0).all(1)
    # Get list of row indexes from dataframe containing only rows to drop
    rows_to_drop = list(df_to_clean[rows_to_drop_mask].index.values)
    # Drop selected rows from dataframe to clean
    df_cleaned = df_to_clean.drop(rows_to_drop)

    # Return new dataframe only containing correctly gathered data
    return df_cleaned

# Normalize dataset using min-max normalization
def normalize_data(df_to_normalize):
    # Create copy of dataframe given as parameter
    df_to_normalize = df_to_normalize.copy()
    # Dataframe to be modified with normalized data
    df_normalized = df_to_normalize.copy()

    # Get list of feature names that need to be normalized
    features_to_normalize = df_to_normalize.columns[TO_NORMALIZE_FEATURES_RANGE]
    
    # Iterate over each feature that needs to be normalized
    for feature in features_to_normalize:
        # Get min and max values from feature column
        max_value = df_to_normalize[feature].max()
        min_value = df_to_normalize[feature].min()
        # If feature column only contains 0s, store 0.0
        if (max_value - min_value) == 0:
            df_normalized[feature] = float(0)
        # If feature column does not contain 0s only, replace each entry with normalized data
        else:
            df_normalized[feature] = (df_normalized[feature] - min_value) / (max_value - min_value)

    # Return normalized dataframe
    return df_normalized

# Convert string values to numeric (necessary for dt classification)
def map_data(df_to_map):
    # Create copy of dataframe given as parameter
    df_to_map = df_to_map.copy() 
    # Dataframe to be mapped with dictionaries
    df_mapped = df_to_map.copy()

    # Mapping dictionary for string to numeric features
    sex_map = {'Maschile': 0, 'Femminile': 1}
    work_map = {'Manuale': 0, 'Intellettuale': 1}
    label_map = {'Sano': 0, 'Malato': 1}

    # Map parameter dataframe with mapping dictionary
    df_mapped['Sex'] = df_to_map['Sex'].map(sex_map)
    df_mapped['Work'] = df_to_map['Work'].map(work_map)
    df_mapped['Label'] = df_to_map['Label'].map(label_map)

    # Return correctly mapped dataframe
    return df_mapped
        
# Export dataframe list as .csv files
def export_data(df_to_export, df_index, df_type):
    # Match filename and folderpath with type of dataframe being exported
    match df_type:
        case 'air':
            base_name = 'onAir_T'
            base_folder_path = Path(AIR_PROCESSED_DATAPATH)
        case 'paper':
            base_name = 'onPaper_T'
            base_folder_path = Path(PAPER_PROCESSED_DATAPATH)
        case 'ap':
            base_name = 'onAirOnPaper_T'
            base_folder_path = Path(AP_PROCESSED_DATAPATH)

    # Create directory in which to export data if it does not exist already
    base_folder_path.mkdir(parents = True, exist_ok = True)

    # Create filename using file index given as argument to match format of input data
    if (df_index < 9):
        file_no = '0' + str(df_index + 1)
    else:
        file_no = str(df_index + 1)

    # Create full file path by contatenating the base folder path with the full file name
    full_file_name = base_name + file_no + '.csv'
    full_file_path = base_folder_path / full_file_name

    # Export dataframe to the complete file path
    df_to_export.to_csv(full_file_path, index = False)

# Main function
def main():
    # Make three lists containing all corresponding CSV file paths that need to be read
    air_csv_list_raw = sorted(glob.glob(os.path.join(AIR_RAW_DATAPATH, "*.csv")))
    paper_csv_list_raw = sorted(glob.glob(os.path.join(PAPER_RAW_DATAPATH, "*.csv")))
    ap_csv_list_raw = sorted(glob.glob(os.path.join(AP_RAW_DATAPATH, "*.csv")))

    # Read all CSV files from the file lists and store them in corresponding dataframe list
    print("Reading files from raw data folder...")
    air_df_list_raw = [ pd.read_csv(csv, sep = SEP, converters = {'Sex': str.strip,
                                                                  'Work': str.strip,
                                                                  'Label': str.strip}) for csv in air_csv_list_raw ]
    paper_df_list_raw = [ pd.read_csv(csv, sep = SEP, converters = {'Sex': str.strip,
                                                                  'Work': str.strip,
                                                                  'Label': str.strip}) for csv in paper_csv_list_raw ]
    ap_df_list_raw = [ pd.read_csv(csv, sep = SEP, converters = {'Sex': str.strip,
                                                                  'Work': str.strip,
                                                                  'Label': str.strip}) for csv in ap_csv_list_raw ]
    
    # Clean up input data where feature acquisition failed
    print("Cleaning datasets...")
    air_df_list_cleaned = [ clean_data(df) for df in air_df_list_raw ]
    paper_df_list_cleaned = [ clean_data(df) for df in paper_df_list_raw ]
    ap_df_list_cleaned = [ clean_data(df) for df in ap_df_list_raw ]

    # Normalize all dataframes from the dataframe lists and store them 
    print("Normalizing datasets...")
    air_df_list_normalized = [ normalize_data(df) for df in air_df_list_cleaned ]
    paper_df_list_normalized = [ normalize_data(df) for df in paper_df_list_cleaned ]
    ap_df_list_normalized = [ normalize_data(df) for df in ap_df_list_cleaned ]

    # Map string features to numeric features
    print("Mapping datasets...")
    air_df_list_mapped = [ map_data(df) for df in air_df_list_normalized ]
    paper_df_list_mapped = [ map_data(df) for df in paper_df_list_normalized ]
    ap_df_list_mapped = [ map_data(df) for df in ap_df_list_normalized ]

    # Export normalized dataframes as CSV to make it accessible from classifiers
    print("Exporting new datasets to processed data folder...")
    [ export_data(df, df_index, 'air') for df_index, df in enumerate(air_df_list_mapped) ]
    [ export_data(df, df_index, 'paper') for df_index, df in enumerate(paper_df_list_mapped) ]
    [ export_data(df, df_index, 'ap') for df_index, df in enumerate(ap_df_list_mapped) ]

    print("Done!")

if __name__ == "__main__":
    main()
