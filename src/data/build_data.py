from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import glob
import os

# Column range holding features that need to be normalized (numeric features)
TO_NORMALIZE_FEATURES_RANGE = np.r_[ 1:87, 88, 90 ]
# Column range holding features that need to be filtered out (could contain only 0s)
TO_CLEAN_FEATURES_RANGE = np.r_[ 1:86 ]
# Separator character used to read input data
SEP = "," 

# Drop erroneous acquisitions by checking rows with only 0-values
def clean_data(df_to_clean):
    # Create copy of dataframe given as parameter
    df_to_clean = df_to_clean.copy()

    # Create mask for all rows with only 0 values in [TO_CLEAN_FEATURES_RANGE] range
    rows_to_drop_mask = df_to_clean.iloc[:, TO_CLEAN_FEATURES_RANGE].eq(0).all(1)
    # Get list of indexes for rows that match mask
    rows_to_drop = list(df_to_clean[rows_to_drop_mask].index.values)
    # Drop selected rows from raw dataframe
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
        # If feature column does not contain 0s only, replace each entry with normalized value
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

    # Mapping dictionaries for string to numeric features
    sex_map = {'Maschile': 0, 'Femminile': 1}
    work_map = {'Manuale': 0, 'Intellettuale': 1}
    label_map = {'Sano': 0, 'Malato': 1}

    # Map dataframe with mapping dictionaries
    df_mapped['Sex'] = df_to_map['Sex'].map(sex_map)
    df_mapped['Work'] = df_to_map['Work'].map(work_map)
    df_mapped['Label'] = df_to_map['Label'].map(label_map)

    # Return correctly mapped dataframe
    return df_mapped

# Complete set of preprocessing operations
def cleanse_data(input_to_cleanse, output_path):
    # Read input data
    df_raw = pd.read_csv(input_to_cleanse, sep=SEP, converters={'Sex': str.strip,
                                                     'Work': str.strip,
                                                     'Label': str.strip})
    # Clean input data
    df_cleaned = clean_data(df_raw) 
    # Normalize input data
    df_normalized = normalize_data(df_cleaned)
    # Map input data
    df_mapped = map_data(df_normalized)
    # Export preprocessed data
    export_data(df_mapped, output_path)

# Export df as .csv file
def export_data(df_to_export, export_path):
    # Export file using path specified with -o flag
    df_to_export.to_csv(export_path, index = False)

# Main function
def main():
    # Set up parser
    parser = argparse.ArgumentParser(description="Python script used to preprocess raw data for classification",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # Add possible cli arguments to parser
    parser.add_argument("-i", help = "Input file to read in .csv format")
    parser.add_argument("-o", help = "Output file to write in .csv format")
    # Parse cli arguments and store them in variable 'args'
    args = parser.parse_args()
    args = vars(args)
    # Store cli arguments
    input_path = args['i']
    output_path = args['o']
    
    # Create output directory if it does not exist
    output_file_dir = Path(os.path.dirname(output_path))
    output_file_dir.mkdir(parents=True, exist_ok=True)
    
    # Check whether input and output arguments are file or directory
    input_is_file = os.path.isfile(input_path)
    input_is_dir = os.path.isdir(input_path)
    output_is_dir = os.path.isdir(output_path)

    # If given -i argument is a file, preprocess it
    if (input_is_file and not(output_is_dir)):
        # Define input/output variables to use as function arguments
        input_file = input_path
        output_file = output_path

        # Preprocess input and export it to output
        cleanse_data(input_file, output_file)

    # If given -i argument is a directory, preprocess every .csv inside it
    if (input_is_dir and output_is_dir):
        # List of filepaths for each file inside input dir
        input_paths = sorted(glob.glob(os.path.join(input_path, '*.csv'))) 
        # List of filenames only inside input dir
        input_file_names = [ os.path.basename(input_path) for input_path in input_paths]
        # List of filepaths for each file inside output dir
        output_paths = [ os.path.join(output_path, file_name) for file_name in input_file_names ]

        # For each file in input dir, read it, preprocess it and export it
        for input_file, output_file in zip(input_paths, output_paths):
            # Preprocess input and export it to output
            cleanse_data(input_file, output_file)

# Main loop
if __name__ == "__main__":
    main()
