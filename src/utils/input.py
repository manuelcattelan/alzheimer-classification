from collections import defaultdict
import os


def scan_input_to_list(input_root, output_root, input_paths=[], output_paths=[]):
    # scan input root dir (find files and subdirs)
    input_root_content = os.listdir(input_root)

    for content in input_root_content:
        # for each content inside input root dir, build content absolute paths
        input_content_path = os.path.join(input_root, content)
        output_content_path = os.path.join(output_root, content)
        # if content is file in csv format, append absolute paths in path lists
        if os.path.isfile(input_content_path) and os.path.splitext(input_content_path)[1] == ".csv":
            input_paths.append(input_content_path)
            output_paths.append(output_content_path)
        # if content is dir, recursively call this function to scan dir content
        elif os.path.isdir(input_content_path) and not input_content_path.startswith("."):
                # update root input path and root output path for recursive call
                new_input_root = input_content_path
                new_output_root = output_content_path
                scan_input_to_list(new_input_root, new_output_root, input_paths, output_paths)

    return sorted(input_paths), sorted(output_paths)


def scan_input_to_dict(input_root, output_root, input_paths=defaultdict(list), output_paths=defaultdict(list)):
    # scan input root dir (find files and subdirs)
    input_root_content = os.listdir(input_root)

    for content in input_root_content:
        # for each content inside input root dir, build content absolute paths
        input_content_path = os.path.join(input_root, content)
        output_content_path = os.path.join(output_root, content)
        # if content is file in csv format, append absolute paths in path lists
        if (os.path.isfile(input_content_path) and os.path.splitext(input_content_path)[1] == ".csv"):
            input_paths[input_root].append(input_content_path)
            output_paths[output_root].append(output_content_path)
        # if content is dir, recursively call this function to scan dir content
        elif os.path.isdir(input_content_path) and not input_content_path.startswith("."):
            # update root input path and root output path for recursive call
            new_input_root = input_content_path
            new_output_root = output_content_path
            scan_input_to_dict(new_input_root, new_output_root, input_paths, output_paths)

    return input_paths, output_paths
