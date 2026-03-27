import hydra
from omegaconf import DictConfig
import pickle
from dl_components.convmixer import ConvMixer
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

import os

from os import listdir
from os.path import isfile, join
import yaml
from dl_components.pl_wrappers import Architecture_PL
from sklearn.metrics import (
    roc_auc_score,
)
import pandas as pd


def calc_auroc_per_ds_for_violation(
    X, training_labels, model=None, ensemble_type="model", graph_type="WCG"
):
    # swap methods to the back according to dataloader
    X = X.transpose(0, 2, 3, 4, 5, 1)

    pred_stack = []
    for x in range(training_labels.shape[0]):
        if ensemble_type == "model":
            pred = model.model(torch.Tensor(X[x])).detach().numpy()
        elif ensemble_type == "mean":
            pred = X[x].mean(axis=-1)
        else:
            pred = X[x, :, :, :, :, ensemble_type]
        
        if graph_type == "WCG":
            pred_stack.append(
                roc_auc_score(
                    (training_labels[x] != 0).flatten(),
                    pred.flatten(),
                )
            )
            
        elif graph_type == "SG":
            print(training_labels[x].shape)
            pred_stack.append(
                roc_auc_score(
                    (training_labels[x] != 0).max(axis=3).flatten(),
                    pred.max(axis=3).flatten(),
                )
            )
    return pred_stack


def extract_best_methods_from_path(path, method_selection=None):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    configs = []
    results = []

    for folder in folders:
        config_path = os.path.join(path, folder, "config.yaml")
        metrics_path = os.path.join(path, folder, "version_0", "metrics.csv")
        if os.path.exists(config_path) and os.path.exists(metrics_path):
            with open(config_path, "r") as f:
                configs.append(yaml.safe_load(f))
            results.append(pd.read_csv(metrics_path))

    configs_df = pd.json_normalize(configs, sep=".")
    configs_df["base_model._target_"] = (
        configs_df["base_model._target_"].str.split(".").str[-1]
    )

    val_aurocs = [df["val_AUROC"] for df in results if "val_AUROC" in df.columns]
    val_aurocs_df = pd.concat(val_aurocs, axis=1)
    val_aurocs_df.columns = [i for i in range(len(val_aurocs_df.columns))]

    stack = []
    evaluate = method_selection if method_selection else configs_df["base_model._target_"].unique()
    for x in configs_df["data.train_ds_path"].unique():
        for y in evaluate:
            c1 = configs_df["data.train_ds_path"] == x
            c2 = configs_df["base_model._target_"] == y

            select = val_aurocs_df[configs_df[c1 & c2].index.values]
            select = select.max().sort_values(ascending=False).head(1)

            stack.append([x.split("/")[-1], y, select.index[0], select.values[0]])
    res_stable = pd.DataFrame(stack, columns=["data", "model", "index", "val_AUROC"])
    
    # Select the items in folders that match the index column of res_stable
    selected_runs = [folders[int(idx)] for idx in res_stable["index"]]

    return res_stable, selected_runs




def sanity_check_model(model, X_eval_b, testing_labels_b):
    # Short test for proper loading.
    X_r = np.concatenate(X_eval_b)
    d = X_r.shape
    X_r = torch.tensor(
        X_r.transpose(0, 2, 3, 4, 5, 1), dtype=torch.float32
    ).reshape(-1, d[-2], d[-2], d[-1], d[1])
    Y_r = np.concatenate(testing_labels_b)
    Y_r = torch.tensor(Y_r, dtype=torch.float32).reshape(-1, d[-2], d[-2], d[-1])
    pred = model.model(X_r[:200]).detach().numpy()
    print(
        "AUROC test with dataloader transform:",
        roc_auc_score((Y_r[:200] != 0).flatten(), pred.flatten()),
    )


def load_model_from_path_and_predict_violations(violation_stack, path, selected_run, X_eval, testing_labels, X_names_eval):
    # Low we run through every violation and dataset and calculate the individual AUROC.
    # This is consistent with the results from the main experiment and serves as a agnostic approach as it likely overreports results

    print("Loading model: ", )
    onlyfiles = [f for f in listdir(path + selected_run) if isfile(join(path + selected_run, f))]
    p = path + selected_run + "/" +  [x for x in onlyfiles if "epoch=" in x][0]
    print(p)
    model = Architecture_PL.load_from_checkpoint(p).to("cpu")
    sanity_check_model(model, X_eval, testing_labels)
    for n in range(0, len(X_eval)):
        print("Using model for prediction.")
        pred_stack = calc_auroc_per_ds_for_violation(
            X_eval[n], testing_labels[n], model=model, ensemble_type="model"
        )
        violation_stack[X_names_eval[n].values[0][0]] = pred_stack

    return violation_stack


@hydra.main(
    version_base="1.3", config_path="config", config_name="7_predict_with_methods.yaml"
)
def main(cfg: DictConfig):
    print(cfg)
    
    
    
    # Setting up everything before actual processing:
    path = cfg.p + cfg.model_p
    
    out_p = cfg.p + cfg.out_folder + "/"
    if not os.path.exists(out_p):
        os.makedirs(out_p)

    # Load the datasets
    # Big
    (
        X_eval_b,
        X_names_eval_b,
        method_order_eval_b,
        testing_labels_b,
        mapping_eval_b,
    ) = pickle.load(open(cfg.p + cfg.big_p, "rb"))
    # Small
    (
        X_eval_s,
        X_names_eval_s,
        method_order_eval_s,
        testing_labels_s,
        mapping_eval_s,
    ) = pickle.load(open(cfg.p + cfg.small_p, "rb"))
    
    
    
    print(X_eval_b.shape)
    print(X_names_eval_b.shape)
    print(method_order_eval_b)
    print(len(testing_labels_b))
    

    print(testing_labels_b[0].shape)
    exit()
    
    # Restrict if specified for debugging purposes mostly.
    if cfg.restrict_violations is not None:
        print("Restricting to", cfg.restrict_violations, "violations.")
        X_eval_b = X_eval_b[: cfg.restrict_violations]
        testing_labels_b = testing_labels_b[: cfg.restrict_violations]
        X_names_eval_b = X_names_eval_b[: cfg.restrict_violations]
        X_eval_s = X_eval_s[: cfg.restrict_violations]
        testing_labels_s = testing_labels_s[: cfg.restrict_violations]
        X_names_eval_s = X_names_eval_s[: cfg.restrict_violations]

    assert method_order_eval_b == method_order_eval_s, (
        "Method orders do not match between big and small datasets."
    )



    if cfg.method_selection in ["SimpleLinear", "SimpleMLP", "ConvMixer"]:
        res_table, selected_runs = extract_best_methods_from_path(path)
        res_table = res_table[res_table["model"] == cfg.method_selection].sort_values("data")
        selected_runs = [selected_runs[i] for i in res_table.index]
        print(res_table)
        print(selected_runs)

        
        violation_stack = {}
        violation_stack = load_model_from_path_and_predict_violations(
            violation_stack, path, selected_runs[0], X_eval_b, testing_labels_b, X_names_eval_b
        )

        violation_stack = load_model_from_path_and_predict_violations(
            violation_stack, path, selected_runs[1], X_eval_s, testing_labels_s, X_names_eval_s
        )
      
     
    else:
        print("Running through the violations:")
        violation_stack = {}
        for n in range(0, len(X_eval_s)):
            if cfg.method_selection in ["Mean"]:
                print("Using mean for prediction.")
                pred_stack = calc_auroc_per_ds_for_violation(
                    X_eval_s[n], testing_labels_s[n], model=None, ensemble_type="mean", graph_type=cfg.graph_type
                )
            else:
                print(
                    "Attempting to interpret the ensemble type as index for the model class"
                )
                pred_stack = calc_auroc_per_ds_for_violation(
                    X_eval_s[n],
                    testing_labels_s[n],
                    model=None,
                    ensemble_type=int(cfg.method_selection), 
                    graph_type=cfg.graph_type
                )
            violation_stack[X_names_eval_s[n].values[0][0]] = pred_stack
            if cfg.method_selection in ["Mean"]:
                print("Using mean for prediction.")
                pred_stack = calc_auroc_per_ds_for_violation(
                    X_eval_b[n], testing_labels_b[n], model=None, ensemble_type="mean", graph_type=cfg.graph_type
                )
            else:
                print(
                    "Attempting to interpret the ensemble type as index for the model class"
                )
                pred_stack = calc_auroc_per_ds_for_violation(
                    X_eval_b[n],
                    testing_labels_b[n],
                    model=None,
                    ensemble_type=int(cfg.method_selection),
                    graph_type=cfg.graph_type
                )
            violation_stack[X_names_eval_b[n].values[0][0]] = pred_stack
            
            
            
    print("Saving....")
    print(len(violation_stack.keys()))
    if cfg.method_selection in  ["Mean", "SimpleLinear", "SimpleMLP", "ConvMixer"]:
        pickle.dump(
            violation_stack, open(out_p + cfg.method_selection +  cfg.graph_type + ".p", "wb")
        )
    else:        
        pickle.dump(
            violation_stack,
            open(
                out_p + method_order_eval_s[cfg.method_selection]+  cfg.graph_type  + ".p", "wb"
            ),
        )

print("Done.")


if __name__ == "__main__":
    main()
