import os


def build_path(
        root_input_path,
        root_output_path,
        content_input_paths,
        content_output_paths
        ):
    # Retrieve root_input_path content (files and subdirectories)
    root_input_content = os.listdir(root_input_path)

    # For each content (file and subdirectory) inside root_input_path:
    for content_path in sorted(root_input_content):
        # Build input/output paths associated with content_path:
        # content_input_path holds the absolute path to the content
        # in the filesystem.
        # content_output_path holds the absolute path to the file in which
        # results (preprocessed data/classification results) obtained from
        # the content will be stored.
        content_input_path = os.path.join(root_input_path, content_path)
        content_output_path = os.path.join(root_output_path, content_path)

        # If content_path points to an existing file with csv extension,
        # append its absolute paths to the corresponding path lists.
        if (os.path.isfile(content_input_path)
                and os.path.splitext(content_input_path)[1] == ".csv"):
            content_input_paths[root_input_path].append(content_input_path)
            content_output_paths[root_output_path].append(content_output_path)

        # If content_path points to an existing visible directory,
        # use its absolute paths as new root paths for a recursive
        # call to this function.
        elif (os.path.isdir(content_input_path)
                and not content_input_path.startswith(".")):
            new_root_input_path = content_input_path
            new_root_output_path = content_output_path
            # content_input_paths and content_input_paths are given
            # as arguments to the recursive function call in order
            # to keep track of all files with csv extension found
            # while traversing the initial root_input_path (args.input).
            build_path(
                    new_root_input_path,
                    new_root_output_path,
                    content_input_paths,
                    content_output_paths
                    )

    return content_input_paths, content_output_paths
