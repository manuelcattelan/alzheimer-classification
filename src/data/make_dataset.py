from src.utils.preprocessing import map_data
from src.utils.preprocessing import clean_data
from src.utils.preprocessing import run_preprocessing
from src.utils.input import input_scan_list
import argparse
import os

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

    # store parsed arguments
    args = parser.parse_args()
    input_path = args.i
    output_path = args.o

    # if input argument is not an existing file or directory, raise exception
    if (not(os.path.isfile(input_path)) and not(os.path.isdir(input_path))):
        raise ValueError(str(input_path) + ' is neither an existing file nor directory')
 
    # get output extension to check output path validity later
    # if input is file -> output must be file with csv extension
    # if input is dir  -> output must be dir without any extension
    output_path_extension = (os.path.splitext(output_path))[1]

    # check if input argument points to file
    if (os.path.isfile(input_path)):
        # if output argument is not a valid path to csv file
        if (output_path_extension != '.csv'):
            raise ValueError(str(output_path) + ' is not a valid csv file path' )

        # preprocess input file
        run_preprocessing(input_path, output_path)

    # check if input argument points to directory
    if (os.path.isdir(input_path)):
        # if output argument is not a valid path to directory
        if (output_path_extension != ''):
            raise ValueError(str(output_path) + ' is not a valid directory path')

        # look for any csv file inside input path and store corresponding paths
        input_paths, output_paths = input_scan_list(input_path, output_path)
        # preprocess each file inside input dir
        for input_path, output_path in zip(input_paths, output_paths):
            run_preprocessing(input_path, output_path)

if __name__ == '__main__':
    main()
