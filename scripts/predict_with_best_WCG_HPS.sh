#!/bin/bash

# --- START MODIFICATION ---

data_base_dir="${1:-/home/datasets4/stein/robust_exp/data_release/}" # First argument: base data directory
output_path="${2:-../ensemble_experiment/ensemble_eval_results/}" # Second argument: output path
skip_first_n="${3:-0}" # Third argument: number of data_paths to skip (default: 0)
specific_dataset="${4:-}" # Fourth argument: optional specific dataset name

# Check if the base data directory exists
if [[ ! -d "$data_base_dir" ]]; then
    echo "Error: Data directory '$data_base_dir' not found. Please create it or adjust the path." >&2
    exit 1
fi

# Initialize an empty array to store data path names
declare -a data_paths

if [[ -n "$specific_dataset" ]]; then
    # If a specific dataset is provided, use only that one
    if [[ -d "$data_base_dir/$specific_dataset" ]]; then
        data_paths=("$specific_dataset")
        echo "Using provided dataset: $specific_dataset"
    else
        echo "Error: Dataset '$specific_dataset' not found in '$data_base_dir'." >&2
        exit 1
    fi
else
    # Find all subdirectories within data_base_dir
    # The trailing '/' ensures that only directories are matched by the glob
    for d in "$data_base_dir"/*/; do
        # Check if 'd' is actually a directory (to handle cases like broken symlinks if any)
        if [[ -d "$d" ]]; then
            # Extract just the folder name (e.g., "dataset_A" from "../data/dataset_A/")
            folder_name=$(basename "$d")
            data_paths+=("$folder_name")
        fi
    done

    # Check if any datasets were found
    if [ ${#data_paths[@]} -eq 0 ]; then
        echo "No subdirectories found in '$data_base_dir'. No experiments to run." >&2
        exit 0 # Exit gracefully if no data is found
    fi

    echo "Found datasets to process: ${data_paths[*]}"
fi

# Skip the first n data_paths if skip_first_n is set
if [[ "$skip_first_n" -gt 0 ]]; then
    if [[ "$skip_first_n" -ge ${#data_paths[@]} ]]; then
        echo "Warning: skip_first_n ($skip_first_n) is greater than or equal to the number of datasets (${#data_paths[@]}). No datasets to process." >&2
        exit 0
    fi
    echo "Skipping first $skip_first_n dataset(s)"
    data_paths=("${data_paths[@]:$skip_first_n}")
    echo "Remaining datasets to process: ${data_paths[*]}"
fi

# --- END MODIFICATION ---

cd ../../cd_zoo

cmd="python benchmark.py -m save=True save_predictions=True method="
cmd2="data_path=$data_base_dir"      # Use the data_base_dir variable
cmd3="which_dataset='range(0,40)'  save_path=$output_path"

methods=(
#"direct_crosscorr"
#"varlingam " #  we skip this as this HP is also the best for INST
#"var method.base_on=coefficients"
#"pcmci ci_test=RobustParCorr"
#"pcmciplus ci_test=RobustParCorr method.reset_lagged_links=False"
#"dynotears" # we skip this as this HP is also the best for INST
#"ntsnotears method.h_tol=1e-60 method.rho_max=1e+16 method.lambda1=0.005 method.lambda2=0.01"
#"cp method.architecture=transformer"
#"fpcmci  # Same
"svarrfci ci_test=RobustParCorr"
)




for data_path in "${data_paths[@]}"; do # data_path will now be just the folder name, e.g., "dataset_A"
    echo "Processing dataset: $data_path"
    for method in "${methods[@]}"; do

        if [[ "$data_path" == *"big"* ]]; then
            cmd4=" method.max_lag=4" 
        fi 
        if [[ "$data_path" == *"small"* ]]; then
            cmd4=" method.max_lag=3" 
        fi
        # The data_path passed to run_degradation_experiment.py will be ../data/dataset_A (etc.)
        echo  "$cmd$method $cmd2$data_path $cmd3$cmd4" 
        eval "$cmd$method $cmd2$data_path $cmd3$cmd4"  
    done
done

wait
echo "Done"