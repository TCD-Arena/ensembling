#!/bin/bash

# Define source and destination paths
SOURCE_DIR="/home/datasets4/stein/robust_exp/release_ensemble_preds/train/WCG/"
DEST_DIR="/home/datasets4/stein/robust_exp/release_ensemble_preds/test_corrected/WCG/"
METHOD="svarrfci" # or "fpcmci"

# Accept start and end indices as arguments (default to 0 and 67)
START_INDEX=${1:-0}
END_INDEX=${2:-67}

echo "Processing experiment folders from index $START_INDEX to $END_INDEX"

# Loop through each experiment folder in SOURCE_DIR
counter=0
for experiment_folder in "$SOURCE_DIR"*/; do
    # Skip folders before START_INDEX
    if [ $counter -lt $START_INDEX ]; then
        ((counter++))
        continue
    fi
    
    # Stop processing after END_INDEX
    if [ $counter -ge $END_INDEX ]; then
        break
    fi
    
    # Extract the experiment folder name
    exp_name=$(basename "$experiment_folder")
    
    # Check if the specified METHOD folder exists in this experiment folder
    if [ -d "${experiment_folder}${METHOD}" ]; then
        # Check if DEST_dir exists in the corresponding destination experiment folder
        dest_path="${DEST_DIR}${exp_name}/"
        
        if [ -d "$dest_path" ]; then
            echo "Copying ${METHOD} from $exp_name to ${dest_path}"
            cp -r "${experiment_folder}${METHOD}" "$dest_path"
        else
            echo "Warning: DEST_dir not found in ${DEST_DIR}${exp_name}/"
        fi
    else
        echo "Warning: ${METHOD} folder not found in $exp_name"
    fi
    
    # Increment counter
    ((counter++))
done

echo "Copy operation completed!"


