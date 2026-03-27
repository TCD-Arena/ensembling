#!/bin/bash

# This script counts occurrences of a specific method across all dataset folders
# Usage: ./count_method_across_datasets.sh <path_to_directory> <method_name>
# Example: ./count_method_across_datasets.sh /home/user/datasets pcmci
# Example: ./count_method_across_datasets.sh ../rename_after_generation varlingam

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide both path and method name as arguments"
    echo "Usage: $0 <path_to_directory> <method_name>"
    echo "Example: $0 /home/user/datasets pcmci"
    echo "Example: $0 ../rename_after_generation varlingam"
    exit 1
fi

TARGET_PATH="$1"
METHOD_NAME="$2"

# Check if the provided path exists
if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: Directory '$TARGET_PATH' does not exist"
    exit 1
fi

echo "Counting occurrences of method '$METHOD_NAME' across all datasets in: $TARGET_PATH"
echo "========================================================================"

# Initialize total counter
total_count=0

# Find all subdirectories in the target path and count method occurrences
find "$TARGET_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dataset_dir; do
    dataset_name=$(basename "$dataset_dir")
    
    # Check if the method subfolder exists in this dataset
    method_dir="$dataset_dir/$METHOD_NAME"
    
    if [ -d "$method_dir" ]; then
        # Count subdirectories in the method folder
        method_count=$(find "$method_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        printf "%-40s %s subfolders\n" "$dataset_name:" "$method_count"
        total_count=$((total_count + method_count))
    else
        printf "%-40s %s\n" "$dataset_name:" "Method not found"
    fi
done

echo "========================================================================"
echo "Total subfolders for method '$METHOD_NAME' across all datasets: $total_count"
