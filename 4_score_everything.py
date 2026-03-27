import pickle
from tkinter import INSERT
import pandas as pd
import numpy as np
import torch
import sys
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from os import listdir
from os.path import isfile, join


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


# Add project root to path to allow imports
if __name__ == "__main__" or "hydra" in sys.modules:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.dirname(__file__))

from cd_zoo.tools.scoring_tools import score





def clip_and_normalize( X):
    """
    Clip values to 2 standard deviations and normalize for each method independently.
    X shape: (batch, method, var, var, lag) or (batch, method, var, var)
    Each method slice is normalized independently over all other dimensions.
    """
    
    X = np.array(X)
    n_methods = X.shape[1]
    X_normalized = np.zeros_like(X)

    # before we do anything, clip occasionnally very large vectors to not skew the training.
    # THis is occasionally happening for models that return coefficients.
    X = X.clip(-10, 10)
    # Process each method independently
    for i in range(n_methods):
        method_slice = X[:, i, ...]
        # Calculate mean and std over all dimensions for this method
        mean = np.mean(method_slice)
        std = np.std(method_slice)
        # Clip to 2 standard deviations
        method_clipped = np.clip(method_slice, mean - 2 * std, mean + 2 * std)
        print("Clipping to:", mean - 2 * std, mean + 2 * std)
        # Normalize the clipped range to [0, 1]
        min_val = np.min(method_clipped)
        max_val = np.max(method_clipped)
        # Avoid division by zero
        range_val = max_val - min_val
        if range_val == 0:
            print("CAREFUL DIVISION BY ZERO IN NORMALIZATION, CHECK YOUR DATA!")
            range_val = 1
        X_normalized[:, i, ...] = (method_clipped - min_val) / range_val
    print(X_normalized.min(), X_normalized.max())
    return X_normalized


def process_and_save_scoring(
    cfg: DictConfig,
    out_meta,
    out_Y,
    out_Y_inst,
    predictions,
    predictions_inst,
    save_path,
):
    ds_list = out_meta["ds"].unique()
    print(ds_list)
    print(f"Found {len(ds_list)} unique datasets.")

    # drop no violation as it is no violation
    ds_list = [d for d in ds_list if "no_violation" not in d]

    if cfg.restrict_to > 0:
        ds_list = ds_list[: cfg.restrict_to]

    print("Using N datasets: ", len(ds_list))

    os.makedirs(save_path, exist_ok=True)
    full_stack = []

    # Example: processing the first dataset found, similar to the notebook "selection = out_meta[out_meta["ds"] == ds[0]]"
    # You might want to iterate over all of them in a real script
    for ds_name in ds_list:
        print(f"Processing dataset: {ds_name}")
        print(
            "Using the following performance score for evaluation:",
            cfg.performance_score,
        )
        selection = out_meta[out_meta["ds"] == ds_name]
        runs = selection["run"].unique()
        stack = []
        for r in runs:
            print(f"Processing run: {r}")
            samples = selection[selection["run"] == r]

            if len(samples) != cfg.experiment.expected_samples:
                print(
                    f"Warning: Expected {cfg.experiment.expected_samples} samples per run, but got {len(samples)} for run {r}"
                )
                continue

            target = out_Y[samples.index]
            preds = predictions[samples.index]
            pred_inst = predictions_inst[samples.index]
            target_inst = out_Y_inst[samples.index]

            # This dataset contains instant links.
            if samples["no_inst"].sum() == 0:
                current_score = score(
                    target,
                    preds,
                    instant_labs=target_inst,
                    instant_preds=pred_inst,
                    per_sample_metrics=("individual" in cfg.performance_score),
                    verbose=cfg.experiment.verbose,
                )
            else:
                current_score = score(
                    target,
                    preds,
                    per_sample_metrics=("individual" in cfg.performance_score),
                    verbose=cfg.experiment.verbose,
                )
            stack.append(current_score.loc[cfg.performance_score])

        out = pd.concat(stack, axis=1).mean(axis=1)
        full_stack.append(out)

    final_out = pd.concat(full_stack, axis=1)
    final_out.columns = ds_list
    print(final_out)
    final_out.to_csv(os.path.join(save_path, "scores.csv"))


@hydra.main(version_base=None, config_path="config", config_name="4_score_everything")
def main(cfg: DictConfig):
    print("Starting pipeline script...")

    # 0. Fix paths for loaded files if they are relative
    # Hydra changes the current working directory.
    # If paths in config are absolute, they are fine.
    # If relative, they might break unless we are careful.

    # 1. Load Data set
    try:  # These are all the predictions we need to score all graph types.
        (
            X_inst_inst,
            out_Y_inst_inst,
            out_meta_inst_inst,
            method_ordering_inst_inst,
        ) = pickle.load(open(cfg.paths.result_p + cfg.size +  "/INST_inst_results.p", "rb"))
        (X_inst_wcg, out_Y_inst_wcg, out_meta_inst_wcg, method_ordering_inst_wcg) = (
            pickle.load(open(cfg.paths.result_p + cfg.size +  "/INST_wcg_results.p", "rb"))
        )
        (X_wcg_wcg, out_Y_wcg_wcg, out_meta_wcg_wcg, method_ordering_wcg_wcg) = (
            pickle.load(open(cfg.paths.result_p + cfg.size +  "/WCG_wcg_results.p", "rb"))
        )

        # Integrity check
        if np.sum(out_meta_inst_wcg.values != out_meta_wcg_wcg.values) == 0:
            print("Meta tables matched successfully.")
        else:
            print("Warning: Meta tables do not match!")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Skipping scoring logic due to missing files.")
        return None

    if cfg.modus == "model_predictions":
        p1 = os.path.join(cfg.paths.model_p, cfg.model, "inst_inst", cfg.size)
        p2 = os.path.join(cfg.paths.model_p, cfg.model, "wcg_wcg", cfg.size)

        # get the pickle path for both cases:
        p1 = [p1 + "/" + f for f in listdir(p1)][0]
        p2 = [p2 + "/" + f for f in listdir(p2)][0]

        print("Using model predictions from: ", p1, p2)

        predictions_inst = np.array(pickle.load(open(p1, "rb"))).mean(
            axis=-1
        )  # squeze last dim
        predictions = np.array(pickle.load(open(p2, "rb")))

        print(predictions_inst.shape, predictions.shape)

        save_path = os.path.join(
            cfg.paths.output_dir, "ensemble_predictions", cfg.model, cfg.size
        )

    elif cfg.modus == "mean_predictions":

                
        
        # we need varlingam, dynotears and fpcmci get the indices from the ordering for them:
        varlingam_idx = method_ordering_inst_wcg.index("varlingam")
        dynotears_idx = method_ordering_inst_wcg.index("dynotears")
        fpcmci_idx = method_ordering_inst_wcg.index("fpcmci")
        # We need to select 3 methods from the inst stack to have all 3 methods
        X_wcg_2 = X_inst_wcg[:, [varlingam_idx, dynotears_idx, fpcmci_idx], :, :]
        X_wcg = torch.concat(
            [torch.tensor(X_wcg_wcg), torch.tensor(X_wcg_2)], dim=1
        ).float()


        if cfg.normalize_predictions:
            X_wcg = clip_and_normalize(X_wcg)
            X_inst_inst = clip_and_normalize(X_inst_inst)
            
            
        predictions = np.array(X_wcg.mean(axis=1))
        predictions_inst = np.array(X_inst_inst.mean(axis=1))

        save_path = os.path.join(
            cfg.paths.output_dir, "mean_predictions", cfg.size
        )

    elif cfg.modus == "consistency_test":
        print(method_ordering_inst_inst, method_ordering_wcg_wcg)
        checker = "direct_crosscorr"
        checker_2 = "dynotears"

        # checker_idx = method_ordering_inst_inst.index(checker)
        checker_idx = method_ordering_wcg_wcg.index(checker)
        checker_idx_2 = method_ordering_inst_inst.index(checker_2)
        predictions_inst = X_inst_inst[:, checker_idx_2]
        predictions = X_wcg_wcg[:, checker_idx]
        print("Running consistency test on :", checker)
        save_path = os.path.join(
            cfg.paths.output_dir,
            "sanity_check",
            cfg.size,
            checker,
        )

    else:
        raise NotImplementedError("modus does not exist.")
        # 2. Run Scoring Logic

    print(predictions.shape, predictions_inst.shape)
    # Now run the processing:
    process_and_save_scoring(
        cfg,
        out_meta_wcg_wcg,
        out_Y_wcg_wcg,
        out_Y_inst_inst,
        predictions,
        predictions_inst,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
