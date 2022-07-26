from collections import defaultdict
import os


def scan_input_dir(input_path,
                   output_path,
                   input_path_list=defaultdict(list),
                   output_path_list=defaultdict(list)):
    # Scan input path and store its content (files/subdirs)
    input_content = os.listdir(input_path)

    # For each content (file/subdir) inside input path
    for content in input_content:
        # Build content absolute paths:
        # content_input_path holds the path to the content in the system
        # content_output_path holds the path to the results from the content
        # This is done in order to keep track of the content
        # paths while recursively traversing the input directory
        content_input_path = os.path.join(input_path, content)
        content_output_path = os.path.join(output_path, content)
        # If content contains path to csv file,
        # append its paths to the corresponding path lists
        if (os.path.isfile(content_input_path)
                and os.path.splitext(content_input_path)[1] == ".csv"):
            input_path_list[input_path].append(content_input_path)
            output_path_list[output_path].append(content_output_path)
        # If content contains path to directory,
        # recursively call this function to scan new directory content
        elif (os.path.isdir(content_input_path)
                and not content_input_path.startswith(".")):
            # Update input_path and output_path to keep track
            # of the new root input_path to traverse
            new_input_path = content_input_path
            new_output_path = content_output_path
            # input_path_list and output_path_list are given
            # as arguments to the recursive function call
            # to keep the list of content paths found
            # inside the root input path updated
            scan_input_dir(new_input_path,
                           new_output_path,
                           input_path_list,
                           output_path_list)

    return input_path_list, output_path_list
