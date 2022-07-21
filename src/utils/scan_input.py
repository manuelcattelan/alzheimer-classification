from collections import defaultdict
import os


def scan_input_dir(
        input_root_path,
        output_root_path,
        input_paths=defaultdict(list),
        output_paths=defaultdict(list)
        ):
    # scan input root dir (find files and subdirs)
    input_root_content = os.listdir(input_root_path)

    # for each content (file or subdir) in input root dir
    for content in input_root_content:
        # build content absolute paths
        # this is done to keep track of the full file paths
        # while traversing the input directory
        input_content_path = os.path.join(input_root_path, content)
        output_content_path = os.path.join(output_root_path, content)
        # if content is file in csv format,
        # append its absolute paths to paths lists
        if (os.path.isfile(input_content_path)
                and os.path.splitext(input_content_path)[1] == ".csv"):
            input_paths[input_root_path].append(input_content_path)
            output_paths[output_root_path].append(output_content_path)
        # if content is dir,
        # recursively call this function to scan new dir content
        elif (os.path.isdir(input_content_path)
                and not input_content_path.startswith(".")):
            # update root input path and root output path
            # to keep track of new root dir path
            new_input_root_path = input_content_path
            new_output_root_path = output_content_path
            # call function recursively and pass as argument
            # bot the input_paths and output_paths list to update
            # with new files found in new directory, if any
            scan_input_dir(
                    new_input_root_path,
                    new_output_root_path,
                    input_paths,
                    output_paths
                    )

    return input_paths, output_paths
