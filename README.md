# Ensembling Experiments Pipeline

This directory contains the complete pipeline for training and evaluating ensemble methods that combine predictions from multiple causal discovery methods. The ensembling approach aims to improve performance over individual methods by leveraging their complementary strengths.

## Overview

The pipeline takes predictions from multiple causal discovery methods (generated in the main cd_zoo experiments) and trains ensemble models to combine these predictions optimally. The ensemble models are evaluated on multiple datasets including synthetic violation datasets and Causal Rivers benchmarks.

## Pipeline Steps

### Step 0: Generate Base Method Predictions

Before running the ensembling pipeline, you need predictions from individual causal discovery methods on all datasets (train/val).

**Run these to predict datasets with the best HP configuration from the main study (Either based on LWCG or on the INSTANTANOUS predictions):**
```bash
cd scripts
./predict_with_best_WCG_HPS.sh /path/to/violation/datasets/ /path/to_prediction/saves/
./predict_with_best_INST_HPS.sh /path/to/violation/datasets/ /path/to_prediction/saves/

```


---

### Step 1: Transform Predictions to Training Sets

**Script:** `1_transform_to_training_set.py`

Transforms the raw prediction files from individual methods into tensor datasets that can be loaded by PyTorch DataLoaders for training ensemble models.

**What it does:**
- Loads predictions from all methods across multiple dataset runs
- Organizes predictions into a structured format (lagged and instantaneous predictions)
- Saves processed data as pickle files for loading during training

**Command to run:**
```bash
python 1_transform_to_training_set.py  data_path=/path/to/violation/datasets/ res_path=/path/to_prediction/saves/INST/

```


### Step 2: Train Ensemble Models

**Script:** `2_train_ensembles.py`

Trains deep learning ensemble architectures (Linear, MLP, Transformer) on the training predictions to learn optimal combination strategies.

**What it does:**
- Loads training data using PyTorch Lightning DataModule
- Trains neural network models to combine method predictions
- Saves model checkpoints and training logs

**Command to run:**
```bash
# TODO: Add your command here
```

**Helper script:** `scripts/train_all_ensembles.sh` - Trains multiple ensemble architectures with different hyperparameters

**Available ensemble architectures:**
- `Linear`: Simple linear combination of predictions
- `MLP`: Multi-layer perceptron 
- `ConvMixer`: Convolutional mixing architecture
- `Transformer`: Transformer-based ensemble

---

### Step 3: Generate Predictions with Trained Ensembles

**Script:** `3_predict.py`

Uses trained ensemble models to generate predictions on test/validation datasets.

**What it does:**
- Loads the best performing ensemble models (based on validation AUROC)
- Generates predictions for all test samples
- Saves predictions for downstream evaluation

**Command to run:**
```bash
# TODO: Add your command here
```

---

### Step 4: Score and Evaluate Ensemble Performance

**Script:** `4_score_everything.py`

Computes performance metrics (AUROC, etc.) for all ensemble predictions and compares against individual methods.

**What it does:**
- Loads predictions from ensemble models and baseline methods
- Calculates performance metrics for each dataset
- Aggregates results across multiple runs and datasets
- Saves scoring results for analysis and visualization

**Command to run:**
```bash
# TODO: Add your command here
```

---

### Step 7: Predict with Best Methods

**Script:** `7_predict_with_best_methods.py`

Generates predictions using the best performing ensemble configurations identified in the validation phase.

**What it does:**
- Selects top-performing ensemble models based on validation metrics
- Generates final test set predictions
- Compares ensemble performance against individual baseline methods
- Computes detailed performance statistics

**Command to run:**
```bash
# TODO: Add your command here
```

---

### Step 9: Apply to Causal Rivers Benchmark

**Script:** `9_use_best_to_predict_cr.py`

Applies the best ensemble models to the Causal Rivers benchmark datasets to evaluate generalization performance.

**What it does:**
- Loads predictions from methods on Causal Rivers datasets
- Applies trained ensemble models to combine these predictions
- Evaluates ensemble performance on real-world benchmarks
- Compares ensemble results against individual method performance

**Command to run:**
```bash
# TODO: Add your command here
```

---

## Additional Utilities

### Mean Ensemble Baseline

**Notebook:** `10_mean_ensemble.ipynb`

Implements and evaluates a simple mean ensemble (averaging predictions) as a baseline comparison.

### Pipeline Orchestration

**Notebooks:**
- `ensemble_pipeline.ipynb`: Interactive pipeline exploration
- `prepare_rivers.ipynb`: Preparation of Causal Rivers data for ensemble evaluation
- `dl_check.ipynb`: Deep learning component validation

### Helper Scripts

Located in `scripts/`:
- `train_all_ensembles.sh`: Batch training of multiple ensemble configurations
- `transform_all_sets.sh`: Batch transformation of datasets
- `predict_with_best_WCG_HPS.sh`: Prediction with best WCG hyperparameters
- `predict_with_best_INST_HPS.sh`: Prediction with best instantaneous hyperparameters  
- `predict_for_causal_rivers.sh`: Generate predictions for Causal Rivers benchmarks

## Configuration

All scripts use Hydra for configuration management. Config files are located in `config/`:
- `1_transform_to_training_set.yaml`
- `2_train_ensembles.yaml`
- `3_predict.yaml`
- `4_score_everything.yaml`
- `base_model/`: Individual ensemble architecture configs

## Output Structure

- `outputs/`: Hydra output logs and run metadata
- Training results and model checkpoints from step 2
- Prediction files from steps 3, 7, and 9
- Performance metrics from step 4

## Dependencies

- PyTorch & PyTorch Lightning
- Hydra for configuration
- pandas, numpy for data processing
- scikit-learn for metrics
- Custom components in `dl_components/`

## Notes

- Ensemble models require predictions from multiple base methods to be effective
- Training is performed separately for different dataset configurations (5 vs 7 variables, different max lags)
- Model selection uses validation AUROC as the primary metric
- The pipeline supports both WCG (weighted causal graph) and INST (instantaneous) graph types



