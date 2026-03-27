# FUll predictions for the ensembling experiments. This script is meant to be run on the cluster, and will run the best WCG and INST hyperparameters for all datasets in the provided data directory. The predictions will be saved to the provided output directory.
./predict_with_best_WCG_HPS.sh /home/datasets4/stein/robust_exp/data_release/ /home/datasets4/stein/robust_exp/release_ensemble_preds/test_corrected/WCG/ 0 
./predict_with_best_INST_HPS.sh /home/datasets4/stein/robust_exp/data_release/ /home/datasets4/stein/robust_exp/release_ensemble_preds/test_corrected/INST/ 0 
./predict_with_best_WCG_HPS.sh /home/datasets4/stein/robust_exp/eval_data_release/ /home/datasets4/stein/robust_exp/release_ensemble_preds/eval_corrected/WCG/ 0 
./predict_with_best_INST_HPS.sh /home/datasets4/stein/robust_exp/eval_data_release/ /home/datasets4/stein/robust_exp/release_ensemble_preds/eval_corrected/INST/ 0 

# Transforms to the training datasets
python 1_transform_to_training_set.py  data_path="/home/datasets4/stein/robust_exp/data_release/" res_path="/home/datasets4/stein/robust_exp/release_ensemble_preds/test_corrected/WCG/"
python 1_transform_to_training_set.py  data_path="/home/datasets4/stein/robust_exp/data_release/" res_path="/home/datasets4/stein/robust_exp/release_ensemble_preds/test_corrected/INST/"
python 1_transform_to_training_set.py  data_path="/home/datasets4/stein/robust_exp/eval_data_release/" res_path="/home/datasets4/stein/robust_exp/release_ensemble_preds/eval_corrected/WCG/"
python 1_transform_to_training_set.py  data_path="/home/datasets4/stein/robust_exp/eval_data_release/" res_path="/home/datasets4/stein/robust_exp/release_ensemble_preds/eval_corrected/INST/"


# For training
./model_grid_search.sh

# For predictions with the best model: 

python 3_predict.py -m predict=True method_selection=SimpleLinear,SimpleTransformer,SimpleMLP modus=wcg_wcg,inst_inst size_selection=5 val_ds_path="/home/datasets4/stein/robust_exp/release_ensemble_results/test_corrected/small/"
python 3_predict.py -m predict=True method_selection=SimpleLinear,SimpleTransformer,SimpleMLP modus=wcg_wcg,inst_inst size_selection=7 val_ds_path="/home/datasets4/stein/robust_exp/release_ensemble_results/test_corrected/big/"







# For rivers


./predict_for_causal_rivers.sh
python 1_extract_results.py  res_path="release_ensemble_rivers/causal_rivers_method_predictions/" out_path="release_ensemble_rivers/"  automated_testing=False ignore_passed=False load_data_hps=False 

generate dataset with rivers.ipynb
python 3_predict.py  predict=True method_selection=SimpleMLP modus=wcg_wcg size_selection=5 out_p="release_ensemble_rivers/ensemble_predictions" rivers_predict=True
