from src.utils.preprocessing import file_preprocessing
from src.utils.preprocessing import dir_preprocessing
import argparse
import os


def main():
    # set up parser and possible arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            )
    parser.add_argument(
            "--input",
            type=str,
            metavar="<input_file/dir>",
            help=("input path of file in .csv format to preprocess or "
                  "directory containing files in .csv format to preprocess"),
            required=True,
            )
    parser.add_argument(
            "--output",
            type=str,
            metavar="<output_file/dir>",
            help=("output path of preprocessed file in .csv format or "
                  "directory containing preprocessed files in .csv format"),
            required=True,
            )

    # store parsed arguments
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # if input argument is not an existing file or directory, raise exception
    if (not os.path.isfile(input_path)
            and not os.path.isdir(input_path)):
        raise ValueError(
                input_path + " does not exist as file or directory"
                )

    # get output extension to check output path validity later
    # if input is file -> output must be file with csv extension
    # if input is dir  -> output must be dir without any extension
    output_path_extension = os.path.splitext(output_path)[1]

    # if input argument points to file
    if os.path.isfile(input_path):
        # if output argument is not a valid path to csv file, raise exception
        if output_path_extension != ".csv":
            raise ValueError(
                    output_path + " is not a valid path to .csv file"
                    )
        # run single file preprocessing
        file_preprocessing(input_path, output_path)

    # if input argument points to directory
    if os.path.isdir(input_path):
        # if output argument is not a valid path to directory, raise exception
        if output_path_extension != "":
            raise ValueError(
                    output_path + " is not a valid directory path"
                    )
        # run dir preprocessing
        dir_preprocessing(input_path, output_path)


if __name__ == "__main__":
    main()
