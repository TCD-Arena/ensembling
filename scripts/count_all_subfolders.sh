#!/bin/bash

# This script calls count_files_in_subfolders.sh for each folder in the specified path
# Usage: ./count_all_subfolders.sh <path_to_directory> [path_to_count_script]
# Example: ./count_all_subfolders.sh /home/user/projects
# Example: ./count_all_subfolders.sh /home/user/projects ./count_files_in_subfolders.sh

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a path as an argument"
    echo "Usage: $0 <path_to_directory> [path_to_count_script]"
    echo "Example: $0 /home/user/projects"
    echo "Example: $0 /home/user/projects ./count_files_in_subfolders.sh"
    exit 1
fi

TARGET_PATH="$1"

# Check if the provided path exists
if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: Directory '$TARGET_PATH' does not exist"
    exit 1
fi

# Default to the count script in the same directory, or use provided path
COUNT_SCRIPT="${2:-$(dirname "$0")/count_files_in_subfolders.sh}"

# Check if the count script exists and is executable
if [ ! -f "$COUNT_SCRIPT" ]; then
    echo "Error: Count script '$COUNT_SCRIPT' does not exist"
    echo "Please provide the correct path to count_files_in_subfolders.sh as the second argument"
    exit 1
fi

if [ ! -x "$COUNT_SCRIPT" ]; then
    echo "Error: Count script '$COUNT_SCRIPT' is not executable"
    echo "Run: chmod +x '$COUNT_SCRIPT'"
    exit 1
fi

echo "Running subfolder count analysis for all directories in: $TARGET_PATH"
echo "========================================================================"

# Find all subdirectories in the target path and run the count script on each
counter=1
find "$TARGET_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r directory; do
    dir_name=$(basename "$directory")
    echo "----------------------------------------"
    echo "[$counter] Dataset: $dir_name"
    # Run the count script on this directory
    "$COUNT_SCRIPT" "$directory"
    echo "----------------------------------------"
    counter=$((counter+1))
done

echo ""
echo "========================================================================"
echo "Analysis complete for all directories in: $TARGET_PATH"
