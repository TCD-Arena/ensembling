import hydra
from omegaconf import DictConfig
import pickle
import torch
import pickle
import pandas as pd
import lightning.pytorch as pl

import os
import numpy as np
from os import listdir
import yaml
from dl_components.pl_wrappers import Architecture_PL
import pandas as pd


def extract_best_methods_from_path(
    path, method_selection=None, modus=None, size_selection=None, cfg=None
):

    if cfg.cache:
        cache_path = os.path.join(cfg.cache_path, "best_runs_cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                configs_df, val_aurocs_df, valid_folders = pickle.load(f)
            print(f"Loaded best runs from cache at {cache_path}")
        else:
            print(f"No cache found at {cache_path}. Extracting best runs from scratch.")
            return None

    else:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        configs = []
        results = []
        valid_folders = []
        # load config and performance table. We use val_AUROC as selection criterium
        for folder in folders:
            config_path = os.path.join(path, folder, "config.yaml")
            metrics_path = os.path.join(path, folder, "version_0", "metrics.csv")
            if os.path.exists(config_path) and os.path.exists(metrics_path):
                with open(config_path, "r") as f:
                    configs.append(yaml.safe_load(f))
                results.append(pd.read_csv(metrics_path))
                valid_folders.append(folder)
            else:
                continue

        configs_df = pd.json_normalize(configs, sep=".")
        configs_df["base_model._target_"] = (
            configs_df["base_model._target_"].str.split(".").str[-1]
        )
        print("RUNS FOUND:")
        print(configs_df["base_model._target_"].value_counts())
        print(configs_df["data.modus"].value_counts())
        print(configs_df["base_model.n_vars"].value_counts())

        val_aurocs = [
            df["val_NegSHD"]
            if "val_NegSHD" in df.columns
            else pd.Series([np.nan] * len(df))
            for df in results
        ]
        val_aurocs_df = pd.concat(val_aurocs, axis=1)
        val_aurocs_df.columns = np.arange(len(val_aurocs_df.columns))
        print("Found runs: ", len(configs_df))

        cache_path = os.path.join(cfg.cache_path, "best_runs_cache.pkl")
        # save to cache:
        os.makedirs(cfg.cache_path, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((configs_df, val_aurocs_df, valid_folders), f)
        print(f"Saved best runs to cache at {cache_path}")

    # Now, for each ensemble type, we select the run with the highest performance
    stack = []
    evaluate = (
        [method_selection]
        if method_selection
        else configs_df["base_model._target_"].unique()
    )
    modus = [modus] if modus else configs_df["data.modus"].unique()
    size = (
        [size_selection] if size_selection else configs_df["base_model.n_vars"].unique()
    )
    for y in evaluate:
        for x in modus:
            for z in size:
                c1 = configs_df["base_model._target_"] == y
                c2 = configs_df["data.modus"] == x
                c3 = configs_df["base_model.n_vars"] == z
                c = c1 & c2 & c3
                if not c.any():
                    print(
                        f"No runs found for method {y}, modus {x}, and size {z}. Skipping."
                    )
                    continue
                # take max from the selected models.
                select = val_aurocs_df[configs_df[c].index.values]
                print(select.max().sort_values(ascending=False))
                select = select.max().sort_values(ascending=False).head(1)
                # select the corresponding folder path for later selection.
                stack.append(
                    [valid_folders[select.index[0]], y, x, z, select.values[0]]
                )
    res_stable = pd.DataFrame(
        stack, columns=["run", "network", "modus", "size", "val_NegSHD"]
    )
    return res_stable


def predict_rivers_and_save(cfg, best_run_id, model_path):
    print(f"Loading best model from run: {best_run_id}")

    # Construct paths
    run_dir = os.path.join(model_path, best_run_id)

    checkpoints = [f for f in listdir(run_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print(f"No checkpoint found in {run_dir}")
        return

    # Prefer 'epoch=' checkpoint which is the better one
    best_ckpt = next((c for c in checkpoints if "epoch=" in c), checkpoints[0])
    ckpt_path = os.path.join(run_dir, best_ckpt)
    print(f"Using checkpoint: {ckpt_path}")

    # Load Model
    model = Architecture_PL.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    X = torch.tensor(pickle.load(open(cfg.rivers_path, "rb"))).to(model.device).float()

    all_preds = model.model(X)
    print(f"Prediction shape: {all_preds.shape}")

    # Save predictions
    out_p = os.path.join(
        cfg.p,
        cfg.out_folder,
        cfg.method_selection,
        cfg.modus,
        *cfg.val_ds_path.split("/")[-2:],
    )
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    out_file = os.path.join(out_p, f"predictions_{best_run_id}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(all_preds.cpu().numpy(), f)
    print(f"Predictions saved to {out_file}")


def predict_and_save(cfg, best_run_id, model_path):
    print(f"Loading best model from run: {best_run_id}")

    # Construct paths
    run_dir = os.path.join(model_path, best_run_id)

    checkpoints = [f for f in listdir(run_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print(f"No checkpoint found in {run_dir}")
        return

    # Prefer 'epoch=' checkpoint which is the better one
    best_ckpt = next((c for c in checkpoints if "epoch=" in c), checkpoints[0])
    ckpt_path = os.path.join(run_dir, best_ckpt)
    print(f"Using checkpoint: {ckpt_path}")

    # Load Model
    model = Architecture_PL.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    # Load Data
    print("Loading Data Module...")
    print(cfg.val_ds_path)
    from dl_components.pl_wrappers import GeneratorDataModule

    dm = GeneratorDataModule(
        train_ds_path=cfg.val_ds_path,  # We dont need the training anyways.
        val_ds_path=cfg.val_ds_path,
        batch_size=32,
        modus=cfg.modus,
        val_percentage_of_samples_to_use=1.0,  # use all data for prediction
        normalize_input=cfg.normalize_input,
    )
    dm.setup(stage="predict")

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

    # Predict
    print("Running prediction...")
    predictions = trainer.predict(model, datamodule=dm)

    # Concatenate predictions if needed
    all_preds = torch.cat(predictions)
    print(f"Prediction shape: {all_preds.shape}")

    # Save predictions
    out_p = os.path.join(
        cfg.p,
        cfg.out_folder,
        cfg.method_selection,
        cfg.modus,
        *cfg.val_ds_path.split("/")[-2:],
    )
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    out_file = os.path.join(out_p, f"predictions_{best_run_id}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(all_preds, f)
    print(f"Predictions saved to {out_file}")


@hydra.main(version_base="1.3", config_path="config", config_name="3_predict.yaml")
def main(cfg: DictConfig):
    print(cfg)

    # Setting up everything before actual processing:
    path = cfg.p + cfg.model_p + "/"
    out_p = cfg.p + cfg.out_folder + "/"
    if not os.path.exists(out_p):
        os.makedirs(out_p)

    if cfg.method_selection in ["SimpleLinear", "SimpleMLP", "SimpleTransformer"]:
        res_table = extract_best_methods_from_path(
            path,
            method_selection=cfg.method_selection,
            modus=cfg.modus,
            size_selection=cfg.size_selection,
            cfg=cfg,
        )
        print("Best model found:")
        print(res_table)
        best_run = res_table.iloc[0]["run"]
        best_run_id = str(best_run)

        if cfg.rivers_predict:
            predict_rivers_and_save(cfg, best_run_id, path)

        elif not res_table.empty and cfg.predict:
            # Call the prediction function
            predict_and_save(cfg, best_run_id, path)

        else:
            print(
                "No valid runs found for the specified criteria. Please check your configurations and the available runs."
            )


if __name__ == "__main__":
    main()
