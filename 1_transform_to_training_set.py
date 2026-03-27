import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import hydra
from omegaconf import DictConfig
import pickle
import os
import time


def check_folder_structure(p):
    """
    Quickly returns ds without 40 subfolders to confirm that the folder structure is correct.
    """
    mypath = p
    onlyfiles = [mypath + f for f in listdir(mypath) if not isfile(join(mypath, f))]
    for x in onlyfiles: 
        if len(listdir(x)) != 40:
            print("ISSUE:", len(listdir(x)), "runs found in: ", x, "correct?")

def load_preds(path, ignore_ds=[], only_ds=None):
    """
    Loads full preds in list form for easier processing.
    """
    dataset = []
    inst_ds = []
    meta = []
    ds = [f for f in listdir(path) if not isfile(join(path, f))]
    print("Datasets found:", ds)
    # filter all datasets that have any  string in cfg.ignore_ds in their name:
    ds = [d for d in ds if all(y not in d for y in ignore_ds)]    
    print("Datasets after ignoring some:", ds)
    ds = [d for d in ds if (only_ds is None or any(y in d for y in only_ds))]
    print("Datasets after restricting :", ds,len(ds))

    for d in ds:
        # Get all available methods: 
        methods = [f for f in listdir(join(path, d)) if not isfile(join(path, d, f))]
        print(f"Methods found in {d}:", methods)
        for method in methods:
            runs = [f for f in listdir(join(path, d, method)) if not isfile(join(path, d, method, f))]
            print(f"Runs found in {d}/{method}:", len(runs))
            for run in runs:
                p = join(path, d, method, run)
                if not isfile(join(p, "preds.p")):
                    print("MISSING preds.p in:", p)
                else:
                    # Load predictions and metadata
                    lagged,inst = pickle.load(open(p + "/preds.p", "rb"))
                    predicted_data_path = pd.read_csv(p + "/scoring.csv", index_col=0).T["path"].values[0].split("/")[-1] 
                    for n,item in enumerate(lagged):
                        meta.append([d,method,run,predicted_data_path,n])
                        dataset.append(item)
                        if isinstance(inst, np.ndarray) and len(inst) > n: 
                            inst_ds.append(inst[n])
                        else:
                            inst_ds.append(np.zeros((item.shape[1], item.shape[1]))) 
                    
    return dataset, inst_ds, pd.DataFrame(meta, columns=["ds","method","run","dpath","index"])



def load_labs(data_path,pred_meta):
    """
    Loads full preds
    """

    labels = []
    label_meta = []
    labels_instant = []
    empty_inst = []
    ds = [f for f in listdir(data_path) if not isfile(join(data_path, f))]
    # only load the datasets that are in the predictions.
    preds_ds = pred_meta["ds"].unique()
    ds = [d for d in ds if d in preds_ds]
    print("Datasets after restricting to predictions:", ds)
    print("datasets found in data path:", len(ds))
    counter = 0 
    for d in ds:
        # Get all available runs: 
        runs = [f for f in listdir(join(data_path, d)) if not isfile(join(data_path, d, f))]
        print(f"Runs found in {d}:", len(runs))
        for run in runs:
            p = join(data_path, d, run)
            if not isfile(join(p, "Y.npy")):
                print("MISSING Y.npy in:", p)
            else:
                # Load predictions and metadata
                Y = np.load(p + "/Y.npy")
                for n,item in enumerate(Y):
                    label_meta.append([d,run,n])
                    # Make binary labels (any link exists or not)
                    labels.append((item != 0).sum(axis=0) > 0)
            assert isfile(join(p, "instant_links.npy")), "MISSING instant_links.npy in: " + p + " FIX if occurs."
            Z = np.load(p + "/instant_links.npy")
            for n,item in enumerate(Z):
                # No links. (link_proba == 0) ( we note this in the meta data table for later processing.)
                if Z.sum() == 0:
                    empty_inst.append(counter)
                labels_instant.append((item != 0).sum(axis=0) > 0)
                counter+=1
    meta = pd.DataFrame(label_meta, columns=["ds","run","index"])
    meta["no_inst"] = False
    meta.loc[empty_inst,"no_inst"] = True
    print(meta["no_inst"].sum())
    return labels, labels_instant, meta




@hydra.main(
    version_base="1.3", config_path="config", config_name="1_transform_to_training_set.yaml"
)
def main(cfg: DictConfig):
    """
    Loads and preps training data after running some concistency checks.
    X output: (Batch x Methods x Vars x Vars x Lags), Instant: (Batch x Methods x Vars x Vars)
    Y_output: (Batch x Vars x Vars x Lags), Y_instant: (Batch x Vars x Vars)
    """
    start_time = time.time()

    check_folder_structure(cfg.data_path)
    print("___")

    
    # Stack all predictions into a single list and a metadata dataframe. 
    # (Note we cant group them into a tensor as the dimensions differ)
    print("Loading predictions...")
    preds,inst_preds,meta = load_preds(path=cfg.res_path, ignore_ds=cfg.ignore_ds, only_ds=cfg.restrict)

    # Display for sanity check.
    print("Raw prediction shapes:")
    print(np.stack(preds).shape)
    print(np.stack(inst_preds).shape)

    # Stack all labels into a single list and a metadata dataframe. 
    # (Note we cant group them into a tensor as the dimensions differ)
    print("Loading Labels...")
    labels, labels_instant, label_meta = load_labs(data_path=cfg.data_path, pred_meta=meta)
    print("Raw label shapes:")
    print(np.stack(labels).shape)
    print(np.stack(labels_instant).shape)


    # To make a proper dataset out of this we need to do two things. 
    # First we need to transform the predictions into a consistent format for training.
    # We do this by transforming them into a two tensor (instant and lagged)
    # Next we also do the same for the labels to match. 
    
    # Optimization: Pre-group meta by (dpath, ds, index) for fast lookups
    print("Pre-processing metadata for fast lookups...")
    meta_grouped = meta.groupby(['dpath', 'ds', 'index'])
    
    method_ordering = None
    out_X = []
    out_inst = []
    good_index = []
    total_samples = len(label_meta)
    print(f"Processing {total_samples} samples...")
    
    # Pre-convert to list for faster iteration
    label_meta_values = label_meta[['run', 'ds', 'index']].values
    
    for idx in range(total_samples):
        # Print progress every 1%
        if idx % max(1, total_samples // 100) == 0:
            progress = (idx / total_samples) * 100
            print(f"Progress: {progress:.1f}% ({idx}/{total_samples})")
        
        run, ds, index_val = label_meta_values[idx]
        
        # Fast lookup using pre-grouped data
        method_stack = meta_grouped.get_group((run, ds, index_val)).sort_values(by="method", ascending=True)

        
        # check if we return always the same ordering
        if not method_ordering: 
            method_ordering = method_stack["method"].values.tolist()
        else: 
            assert method_ordering == method_stack["method"].values.tolist(), "Method ordering mismatch!"
        
        # select method ids from the preds list and combine them into a single tensor: 
        method_ids = method_stack.index.tolist()
        out_X.append(np.stack([preds[i] for i in method_ids]))
        
        # Mark all instant preds as they would be zero tensors.
        method_ids = method_stack[~(method_stack["method"].isin(cfg.no_instant_preds))].index.tolist()  
        out_inst.append(np.stack([inst_preds[i] for i in method_ids]))
        good_index.append(idx)
    
    print(f"Progress: 100.0% ({total_samples}/{total_samples})")

    #shorten the meta data table to only hold indices that were kept
    out_meta = label_meta.loc[good_index].reset_index(drop=True)
    
    # stack for batch dimension
    X = np.stack(out_X)
    X_inst = np.stack(out_inst)    
    
    # also only select the proper indices from the label lists
    out_Y = np.stack([labels[i] for i in good_index])
    out_Y_inst = np.stack([labels_instant[i] for i in good_index])

    print(out_meta.head())
    print(out_meta["no_inst"].sum(), "samples with no instant links.")
    print("Final training data shape:", X.shape, X_inst.shape)
    print("Final instant data shape:", out_Y.shape, out_Y_inst.shape)
    
    split, graph = cfg.res_path.split("/")[-3:-1]
    
    path = cfg.out_path + "/" + split +"/"  + cfg.naming   + "/"
    # create path: 
    os.makedirs(path, exist_ok=True)

    pickle.dump(
        (X,out_Y,out_meta,method_ordering),
        open(path +  graph  +"_wcg_results.p", "wb"),
    )
    pickle.dump(
        (X_inst,out_Y_inst,out_meta,[x for x in method_ordering if x not in cfg.no_instant_preds]),
        open(path +  graph  +"_inst_results.p", "wb"),
    )
    
    elapsed_time = time.time() - start_time
    print(f"Done. Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
