# Ensembling Experiments Pipeline

This directory contains the complete pipeline for training and evaluating ensemble methods that combine predictions from multiple causal discovery methods. The ensembling approach aims to improve performance over individual methods by leveraging their complementary strengths.

## Overview

The pipeline takes predictions from multiple causal discovery methods (generated in the main cd_zoo experiments) and trains ensemble models to combine these predictions optimally. The ensemble models are evaluated on multiple datasets including synthetic violation datasets and Causal Rivers benchmarks.


## 👩‍🔬 Using Ensembles

The simplest way to use our ensembles please follow cd_zoo functionality to prepare individual predictions (Step2 below) and transform the predictions into a tensor (You might follow the steps in 6_generate_rivers_dataset.ipynb). YOu can download the weights of the best performing ensembles.

After you have the data and the model weights, you can use the 3_predict.py functionality. 

Notably, depending on your task, you may subselect certain ensembles as we have 12 in total (3 architectures* 5/7 vars * lagged,instantanous effects )


```bash
wget https://github.com/TCD-Arena/ensembling/releases/download/model_weights/best_ensembles.zip
```


## Pipeline Steps


1. Generate training data
2. predict the full training data with best Hyperparameter configuration found in the main experiment
3. Train ensembles
4. select the highest scoring training run and predict with it
5. Evaluate the performance of the prediction.


### Step 0: Generate Training data: 

For training, we leverage the synth_ds_generator of TCD-arena to train an additional dataset that contains all 33 violations: 

```bash
../synthetic_ds_generator/create_all_violations_datasets 123  /path/to/save/train_data_release # We use the seed 123 for training data

```

### Step 2: Generate predictions for all datasets: 

In our study, we precalculated all predictions beforehand with the best performing Hyperparameter configurations (once for instantanous and once for lagged effects) from the main experiment. This can be done by running:

```bash
# Arguments: Path to data, path_to_output_folder, number of datasets to skip
./predict_with_best_WCG_HPS.sh --/data_release/ release_ensemble_preds/test/WCG/ 0 
./predict_with_best_INST_HPS.sh ../data_release/ release_ensemble_preds/test/INST/ 0 
./predict_with_best_WCG_HPS.sh /path/to/save/train_data_release/ release_ensemble_preds/train/WCG/ 0 
./predict_with_best_INST_HPS.sh /path/to/save/train_data_release release_ensemble_preds/train/INST/ 0 
```


### Step 2: Prepare the raw predictions and labels into a training format: 

We transfer into Tensor datasets that we can easily load for training. Notably, we need train different ensembles for the different data sizes to keep the input format consistent:

```bash
python 1_transform_to_training_set.py -m data_path="../data_release/" res_path="release_ensemble_preds/test/WCG/","release_ensemble_preds/test/INST/" ignore_ds="[small]" naming="big" out_path="release_ensemble_tensor_data/"
python 1_transform_to_training_set.py data_path="train_data_release/" res_path="release_ensemble_preds/test/WCG/","release_ensemble_preds/test/INST/" ignore_ds="[big]" naming="small" out_path="release_ensemble_tensor_data/"
python 1_transform_to_training_set.py data_path="train_data_release/" res_path="release_ensemble_preds/train/WCG/","release_ensemble_preds/train/INST/" ignore_ds="[small]" naming="big" out_path="release_ensemble_tensor_data/"
python 1_transform_to_training_set.py data_path="train_data_release/" res_path="release_ensemble_preds/train/WCG/","release_ensemble_preds/train/INST/" ignore_ds="[big]" naming="small" out_path="release_ensemble_tensor_data/"

```


### Step 3: Train Ensemble approaches: 

We run a grid search for all model types. Notably this likely requires a cluster. 
Therefore, we provide the best performing model for each Architecture as a release
We use the cd_zoo_torch environment for training.

```bash
# We document the grid search here:
./scripts/model_grid_search.sh
```

```bash
# A single training run can be executed with (pathing must be configured):
    python 2_train_ensembles.py
```


### Step 4: Use the best performing Ensemble to predict the test data: 

```bash
# A single training run can be executed with (pathing must be configured):
   python 3_predict.py -m predict=True method_selection=SimpleLinear,SimpleTransformer,SimpleMLP modus=wcg_wcg,inst_inst size_selection=7 out_folder=reproduce_performance val_ds_path="path/to/ensemble_experiments/release_ensemble_tensor_data/test_corrected/big/" 
   python 3_predict.py -m predict=True method_selection=SimpleLinear,SimpleTransformer,SimpleMLP modus=wcg_wcg,inst_inst size_selection=7 out_folder=reproduce_performance val_ds_path="path/to/ensemble_experiments/release_ensemble_tensor_data/test_corrected/big/" 
```

### Step 5: Use the predictions to validate performance: 

Note this function is also computing the mean ensemble from the raw tensor predictions and provides functionality to select single methods from the tensor data to control for consistency between the robustness and ensembling experiments

```bash
# A single training run can be executed with (pathing must be configured):
python 4_score_everything.py -m modus="model_predictions" performance_score="SHD individual" model=SimpleMLP restrict_to=-1 size=small normalize_predictions=True

```


### Step 5: Summarize the results:

In 5_produce_ensemble_results.ipynb we export the results for all ensembles along sanity checks and visualizations



### Step 6: Causal Rivers:

We recycle a number of functionalities from other packages here: 
```bash
# Run to generate raw predictions with cd_zoo functionality. Please follow the causal rivers integration in the main cd_zoo repository first
./predict_for_causal_rivers.sh
```

After this, we can generate the summarized results via TCD-Arena functionality

```bash
python 1_extract_results.py  res_path="path_to/causal_rivers_method_predictions/" out_path="rivers_output_path/"  automated_testing=False ignore_passed=False load_data_hps=False 
```

We then generate a simple dataset in 6_generatee_rivers_dataset,ipynb

Next we predict via this dataset with the best ensembles:

```bash
python 3_predict.py -m  predict=True method_selection=SimpleMLP,SimpleLinear,SimpleTransformer modus=wcg_wcg size_selection=5 out_p="release_ensemble_rivers/ensemble_predictions" rivers_predict=True cache=False out_folder="release_rivers_ensemble_performance_reproduce" out_folder=causal_rivers_performance_reproduce
```
Finally, in 7_performance_on_causarivers.ipynb, we evaluate the performance of these predictions



## Notes

- Ensemble models require predictions from multiple base methods to be effective
- Training is performed separately for different dataset configurations (5 vs 7 variables, different max lags)
- Model selection and checkpoint saving rely on validation negative SHD.
- The pipeline supports both WCG (weighted causal graph) and INST (instantaneous) graph types
- This is research code. If you have trouble getting your own ensembles to work, feel free to contact us.



