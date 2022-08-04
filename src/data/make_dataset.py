from src.utils.path import build_path
from src.utils.preprocessing import preprocess_dataframe
from src.utils.preprocessing import export_dataframe
from collections import defaultdict
import pandas as pd
import argparse
import errno
import os


def main():
    # Setup parser to enable command line arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter
            )
    parser.add_argument(
            "--input",
            help=("path to file or directory of files where data "
                  "to preprocess is stored"),
            required=True
            )
    parser.add_argument(
            "--output",
            help=("path to file or directory of files where "
                  "preprocessed data will be stored"),
            required=True
            )
    args = parser.parse_args()

    # Check if provided input argument is valid, meaning:
    # args.input is a path to an existing file, or
    # args.input is a path to an existing directory
    if not os.path.exists(args.input):
        raise FileExistsError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                args.input
                )

    # Get extension from output argument for later validity checks:
    # If args.input is a file -> args.output must end with .csv extension
    # If args.input is a directory -> args.output must end with no extension
    output_extension = os.path.splitext(args.output)[1]

    # Check if provided input argument holds path to existing file
    if os.path.isfile(args.input):
        if output_extension != ".csv":
            raise ValueError(
                    "not a valid path to csv file: '"
                    + args.output
                    + "'"
                    )
        # If all conditions are met:
        # read raw data
        # preprocess raw data
        # export preprocessed data
        df_to_preprocess = pd.read_csv(args.input, sep=";")
        df_preprocessed = preprocess_dataframe(df_to_preprocess)
        export_dataframe(df_preprocessed, args.output)

    # Check if provided input argument holds path to existing directory
    if os.path.isdir(args.input):
        if output_extension != "":
            raise ValueError(
                    "not a valid path to directory: '"
                    + args.output
                    + "'"
                    )
        # If all conditions are met:
        # Build input/output paths for each file/directory inside args.input
        input_paths, output_paths = build_path(
                args.input,
                args.output,
                defaultdict(list),
                defaultdict(list)
                )
        # For each directory inside args.input
        for input_dirpath, output_dirpath in zip(
                sorted(input_paths),
                sorted(output_paths)
                ):
            # For each file inside directory
            for input_filepath, output_filepath in zip(
                    sorted(input_paths[input_dirpath]),
                    sorted(output_paths[output_dirpath])
                    ):
                # read raw data
                # preprocess raw data
                # export preprocessed data
                df_to_preprocess = pd.read_csv(input_filepath, sep=";")
                df_preprocessed = preprocess_dataframe(df_to_preprocess)
                export_dataframe(df_preprocessed, output_filepath)


if __name__ == "__main__":
    main()
