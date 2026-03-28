#!/bin/bash

# For training
# Usage: ./model_grid_search.sh [model_name]
# model_name: linear, mlp, transformer (optional - runs all if not specified)

cd ..

BASE_DIR="/path/to/release_ensemble_tensor_data"

# Parse command line argument
SELECTED_MODEL="${1:-all}"

# Define configurations
declare -A configs=(
    ["small_inst"]="base_model.methods=6 base_model.lag_in=1 base_model.lag_out=1 base_model.n_vars=5 data.modus='inst_inst' data.normalize_input=True,False"
    ["big_inst"]="base_model.methods=6 base_model.lag_in=1 base_model.lag_out=1 base_model.n_vars=7 data.modus='inst_inst' data.normalize_input=True,False"
    ["small_wcg"]="base_model.methods=9 base_model.lag_in=3 base_model.lag_out=3 base_model.n_vars=5 data.modus='wcg_wcg' data.normalize_input=True,False"
    ["big_wcg"]="base_model.methods=9 base_model.lag_in=4 base_model.lag_out=4 base_model.n_vars=7 data.modus='wcg_wcg' data.normalize_input=True,False"
)

# Define dataset sizes for each config
declare -A sizes=(
    ["small_inst"]="small"
    ["big_inst"]="big"
    ["small_wcg"]="small"
    ["big_wcg"]="big"
)

# Define model architectures (split by loss type to avoid unnecessary BCE weight runs with MSE)
declare -A models=(
    ["linear_mse"]="model.loss_type=mse data.batch_size=32,128 model.optimizer_lr=1e-4,1e-2 model.weight_decay=0.01,0.1 base_model=linear"
    ["linear_bce"]="model.loss_type=bce data.batch_size=32,128 model.optimizer_lr=1e-4,1e-2 model.weight_decay=0.01,0.1 model.positive_weight_bce=1.0,2.5,3.0 base_model=linear"
    ["linear_focal"]="model.loss_type=focal data.batch_size=32,128 model.optimizer_lr=1e-4,1e-2 model.weight_decay=0.01,0.1 base_model=linear"
    ["mlp_mse"]='model.loss_type=mse data.batch_size=32,128 model.optimizer_lr=1e-4,1e-2 model.weight_decay=0.01,0.1 base_model=mlp base_model.dropout_rate=0,0.1'
    ["mlp_bce"]='model.loss_type=bce data.batch_size=1024,2048 model.optimizer_lr=1e-4 model.weight_decay=0.01,0.1 model.positive_weight_bce=2.5,1.5 base_model=mlp base_model.dropout_rate=0,0.1'
    ["mlp_focal"]='model.loss_type=focal data.batch_size=1024,2048 model.optimizer_lr=1e-4 model.weight_decay=0.01,0.1 base_model=mlp base_model.dropout_rate=0,0.1'
    ["transformer_mse"]="model.loss_type=mse data.batch_size=128,256 model.optimizer_lr=1e-4 model.weight_decay=0.01,0.1 base_model=transformer base_model.model_dim=256,512 base_model.num_layers=2 base_model.dropout=0,0.1"
    ["transformer_bce"]="model.loss_type=bce data.batch_size=128,256 model.optimizer_lr=1e-4 model.weight_decay=0.01,0.1 base_model=transformer base_model.model_dim=256,512 base_model.num_layers=2 base_model.dropout=0,0.1 model.positive_weight_bce=1.0,2.5"
    ["transformer_focal"]="model.loss_type=focal data.batch_size=128,256 model.optimizer_lr=1e-4 model.weight_decay=0.01,0.1 base_model=transformer base_model.model_dim=256,512 base_model.num_layers=2 base_model.dropout=0,0.1"

)

# Validate selected model
if [[ "$SELECTED_MODEL" != "all" ]]; then
    # Check if any model key starts with the selected model name
    valid=false
    for key in "${!models[@]}"; do
        if [[ "$key" == "${SELECTED_MODEL}_"* || "$key" == "$SELECTED_MODEL" ]]; then
            valid=true
            break
        fi
    done
    if [[ "$valid" == false ]]; then
        echo "Error: Invalid model '$SELECTED_MODEL'. Choose from: linear, mlp, transformer, or all"
        exit 1
    fi
fi

# Loop through all combinations
for config_name in "${!configs[@]}"; do
    size="${sizes[$config_name]}"
    config_args="${configs[$config_name]}"
    
    for model_name in "${!models[@]}"; do
        # Skip if a specific model is selected and this model doesn't match
        if [[ "$SELECTED_MODEL" != "all" ]]; then
            if [[ "$model_name" != "${SELECTED_MODEL}_"* && "$model_name" != "$SELECTED_MODEL" ]]; then
                continue
            fi
        fi
        
        model_args="${models[$model_name]}"
        
        echo python 2_train_ensembles.py -m $model_args $config_args \
            data.train_ds_path="$BASE_DIR/eval_corrected/$size/" \
            data.val_ds_path="$BASE_DIR/test_corrected/$size/" 
        #  eval python 2_train_ensembles.py -m $model_args $config_args \
        #       data.train_ds_path="$BASE_DIR/eval_corrected/$size/" \
        #       data.val_ds_path="$BASE_DIR/test_corrected/$size/" 
    done
done

