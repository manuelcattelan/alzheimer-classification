from src.utils.preprocessing import file_preprocessing
from src.utils.preprocessing import dir_preprocessing
import argparse
import logging
import errno
import os


def main():
    # Set un parser to enable possible arguments from command line
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",
                        help=("path to file or directory of files where "
                              "data to preprocess is stored"),
                        required=True)
    parser.add_argument("--output",
                        help=("path to file or directory of files where "
                              "preprocessed data will be stored"),
                        required=True)
    args = parser.parse_args()

    # Check if provided input argument is valid, meaning:
    # Input argument is an existing file, or
    # Input argument is an existing directory
    if not os.path.exists(args.input):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                args.input)

    # Get extension from output argument for later validity checks:
    # If input is a file -> output must end with .csv extension
    # If input is a directory -> output must end with no extension
    output_arg_extension = os.path.splitext(args.output)[1]

    # Check if provided input argument contains path to file
    if os.path.isfile(args.input):
        if output_arg_extension != ".csv":
            raise ValueError("Not a valid path to csv file: '"
                             + args.output
                             + "'")
        # If everything is OK:
        # run preprocessing on specified input file
        # store preprocessed data on specified output file
        file_preprocessing(args.input, args.output)

    # Check if provided input argument contains path to directory
    if os.path.isdir(args.input):
        if output_arg_extension != "":
            raise ValueError("Not a valid path to directory: '"
                             + args.output
                             + "'")
        # If everything is OK:
        # run preprocessing on specified input directory
        # store preprocessed data on specified output directory
        dir_preprocessing(args.input, args.output)


if __name__ == "__main__":
    main()
