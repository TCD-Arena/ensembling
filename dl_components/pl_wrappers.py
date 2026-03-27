import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as opt
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Metric
from torchmetrics.classification import BinaryAUROC
import torch.nn.functional as F
import sys
sys.path.append('/home/stein/project_repos/tcd_arena')
from cd_zoo.tools.scoring_tools import min_shd


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NormalizedSHD(Metric):
    """
    Normalized Structured Hamming Distance metric.
    Lower is better. Accumulates predictions and targets, then computes optimal SHD at the end.
    """
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Apply sigmoid to convert logits to probabilities
        preds = torch.sigmoid(preds)
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self):
        # Concatenate all accumulated predictions and targets
        preds = torch.cat(self.preds).numpy().flatten()
        targets = torch.cat(self.targets).numpy().flatten()
        
        # Use the min_shd function from scoring_tools
        _, shd_score = min_shd(targets, preds)
        
        # Return negative because lower SHD is better, but we want to maximize the metric
        return torch.tensor(-shd_score, dtype=torch.float32)


class ensemble_data(Dataset):
    """
    WE have 4 modi that we test:

    inst_inst: Uses only instant predictions to predict instant predictions
    wcg_wcg: Uses weighted context gradients to predict weighted context gradients
    joint_wcg: Uses INST+WCG predictions to predict weighted context gradients
    joint_inst: Uses INST+WCG context gradients to predict instant predictions

    Notably as the baseline, we rely on prediction from base methods.
    Depending on the prediction target, the best method configuration changes.
    We therefore predicted all samples with partly two versions of the same method.
    We have to select the correct version here accordingly.
    Generally: If we want to predict WCG we select the WCG version of the method and vice versa.


    Dimensions of the datasets:


    X:                                 WCG INST origin of preds
    wcg_wcg(n_samples, 10, 7, 7, 3) (6 4)
    inst_inst(n_samples, 6, 7, 7) (6 0    )
    joint_wcg(n_samples, 10, 7, 7, 4) (7x3)
    joint_inst(n_samples, 10, 7, 7, 4) (6x4)


    Y:
    wcg_wcg(n_samples, 7, 7, 3)
    inst_inst(n_samples, 7, 7)
    joint_wcg(n_samples, 7, 7, 3)
    joint_inst: (n_samples, 10, 7, 7, 3)


    """

    def __init__(
        self,
        ds_path="single",
        modus="inst_inst",
        percentage_of_samples_to_use=1.0,
        normalize_data=True,
    ):
        self.ds_path = ds_path
        self.modus = modus
        self.normalize_data = normalize_data
        self.percentage_of_samples_to_use = percentage_of_samples_to_use
        self.X, self.Y = self.load()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def _clip_and_normalize(self, X):
        """
        Clip values to 2 standard deviations and normalize for each method independently.
        X shape: (batch, method, var, var, lag) or (batch, method, var, var)
        Each method slice is normalized independently over all other dimensions.
        """
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

    def load(self):
        # We load different dataproduct depending on the model
        if self.modus == "inst_inst":
            (X_inst, out_Y_inst, out_meta_inst, method_ordering_inst) = pickle.load(
                open(self.ds_path + "/INST_inst_results.p", "rb")
            )

            if self.percentage_of_samples_to_use < 1.0:
                n_samples = X_inst.shape[0]
                n_samples_to_use = int(n_samples * self.percentage_of_samples_to_use)
                # seed this properly:
                np.random.seed(42)
                indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
                X_inst = X_inst[indices]
                out_Y_inst = out_Y_inst[indices]

            # Print class distribution for debugging
            pos_ratio = out_Y_inst.mean().item()
            print(f"Positive class ratio: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
            print(f"Suggested positive_weight_bce: {(1-pos_ratio)/pos_ratio:.2f}")

            # Clip to 2 standard deviations and normalize along method axis
            if self.normalize_data:
                X_inst = self._clip_and_normalize(X_inst)
            else:
                X_inst = X_inst.clip(-10, 10)
            return torch.tensor(X_inst).float(), torch.tensor(
                out_Y_inst
            ).float().unsqueeze(-1)

        elif self.modus == "wcg_wcg":
            (X_wcg, out_Y_wcg, out_meta_wcg, method_ordering_wcg) = pickle.load(
                open(self.ds_path + "/WCG_wcg_results.p", "rb")
            )
            (X_wcg_inst, _, out_meta_inst, method_ordering_inst) = pickle.load(
                open(self.ds_path + "/INST_wcg_results.p", "rb")
            )
            # we need varlingam, dynotears and fpcmci get the indices from the ordering for them:
            varlingam_idx = method_ordering_inst.index("varlingam")
            dynotears_idx = method_ordering_inst.index("dynotears")
            fpcmci_idx = method_ordering_inst.index("fpcmci")
            print(method_ordering_wcg)
            # We need to select 3 methods from the inst stack to have all 3 methods
            X_wcg_inst = X_wcg_inst[:, [varlingam_idx, dynotears_idx, fpcmci_idx], :, :]
            X_wcg = torch.concat(
                [torch.tensor(X_wcg), torch.tensor(X_wcg_inst)], dim=1
            ).float()
            if self.percentage_of_samples_to_use < 1.0:
                n_samples = X_wcg.shape[0]
                n_samples_to_use = int(n_samples * self.percentage_of_samples_to_use)
                # seed this properly:
                np.random.seed(42)
                indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
                X_wcg = X_wcg[indices]
                out_Y_wcg = out_Y_wcg[indices]

            # Clip to 2 standard deviations and normalize along method axis
            if self.normalize_data:
                X_wcg = self._clip_and_normalize(X_wcg.numpy())
            else:
                X_wcg = X_wcg.clip(-10, 10)
            return torch.tensor(X_wcg).float(), torch.tensor(out_Y_wcg).float()

        elif self.modus in ["joint_wcg", "joint_inst"]:
            assert False, "Not implemented yet.  adapt it accordingly."
            (X_wcg, out_Y_wcg, out_meta_wcg, method_ordering_wcg) = pickle.load(
                open(self.ds_path + "/eval_corrected_WCGsmall_wcg_results.p", "rb")
            )
            (X_wcg_2, out_Y_inst, out_meta_wcg, method_ordering_wcg) = pickle.load(
                open(self.ds_path + "/eval_corrected_WCGssmall_inst_results.p", "rb")
            )
            (X_inst, out_Y_wcg, out_meta_wcg, method_ordering_inst) = pickle.load(
                open(self.ds_path + "/eval_corrected_INSTsmall_wcg_results.p", "rb")
            )
            (X_inst_2, out_Y_inst, out_meta_wcg, method_ordering_inst) = pickle.load(
                open(self.ds_path + "/eval_corrected_INSTssmall_inst_results.p", "rb")
            )

            if self.modus == "joint_wcg":
                # For these I need to use the 7 wcg methods, extend instant predictions,
                # stack the 3 INST methods on the lag dimension and stack them all in the method dimension

                X_wcg = torch.concat([X_wcg, X_wcg_2.unsqueeze(-1)], dim=-1)
                X_inst = torch.concat([X_inst, X_inst_2.unsqueeze(-1)], dim=-1)
                # we need varlingam, dynotears and fpcmci get the indices from the ordering for them:
                varlingam_idx = method_ordering_inst.index("varlingam")
                dynotears_idx = method_ordering_inst.index("dynotears")
                fpcmci_idx = method_ordering_inst.index("fpcmci")

                # We need to select 3 methods from the inst stack to have all 3 methods
                X_inst = X_inst[:, [varlingam_idx, dynotears_idx, fpcmci_idx], :, :]
                X_wcg = torch.concat([X_wcg, X_inst], dim=1)
                return X_wcg, out_Y_wcg.float()
            else:
                # For these I need to use the 6 inst methods, extend wcg 4 predictions,
                X_wcg = torch.concat([X_wcg, X_wcg_2.unsqueeze(-1)], dim=-1)
                X_inst = torch.concat([X_inst, X_inst_2.unsqueeze(-1)], dim=-1)
                # we need methods that have no instant predictions: crosscorr, cp, var and pcmci:
                crosscorr_idx = method_ordering_inst.index("crosscorr")
                cp_idx = method_ordering_inst.index("cp")
                var_idx = method_ordering_inst.index("var")
                pcmci_idx = method_ordering_inst.index("pcmci")

                # We need to select 3 methods from the inst stack to have all 3 methods
                X_wcg = X_wcg[:, [crosscorr_idx, cp_idx, var_idx, pcmci_idx], :, :]
                X_inst = torch.concat([X_wcg, X_inst], dim=1)
                return X_wcg, out_Y_inst.float()

        else:
            raise ValueError(f"Unknown modus: {self.modus}")


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds_path,
        val_ds_path,
        modus: str = "inst_inst",
        batch_size: int = 32,
        val_percentage_of_samples_to_use=1.0,
        normalize_input: bool = True,
    ):

        super().__init__()
        self.train_ds_path = train_ds_path
        self.val_ds_path = val_ds_path
        self.modus = modus
        self.batch_size = batch_size
        self.normalize_input = normalize_input
        self.val_percentage_of_samples_to_use = val_percentage_of_samples_to_use

    def setup(self, stage):

        self.train_ds = ensemble_data(
            ds_path=self.train_ds_path,
            modus=self.modus,
            percentage_of_samples_to_use=1.0,
            normalize_data=self.normalize_input,
        )
        self.val_ds = ensemble_data(
            ds_path=self.val_ds_path,
            modus=self.modus,
            percentage_of_samples_to_use=self.val_percentage_of_samples_to_use,
            normalize_data=self.normalize_input,
        )
        self.predict_ds = ensemble_data(
            ds_path=self.val_ds_path,
            modus=self.modus,
            percentage_of_samples_to_use=1.0,
            normalize_data=self.normalize_input,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def teardown(self, stage):
        pass


class Architecture_PL(pl.LightningModule):
    def __init__(
        self,
        loss_type="bce",  # Default to BCE for binary classification
        optimizer_lr=1e-4,
        weight_decay=0.01,
        base_model=None,
        positive_weight_bce=1.0,
        focal_alpha=0.25,  # For focal loss
        focal_gamma=2.0,  # For focal loss
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.loss_type = loss_type
        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay
        self.positive_weight_bce = positive_weight_bce
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.model = base_model

        self.loss = self.loss_init()
        self.val_metric = self.val_metrics_init()
        self.train_metric = self.val_metrics_init()

    def val_metrics_init(self):
        return torch.nn.ModuleDict({
            "AUROC": BinaryAUROC(),
            "NegSHD": NormalizedSHD()
        })

    def loss_init(self):
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "bce":
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.positive_weight_bce)
            )
        elif self.loss_type == "focal":
            return FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            return None

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_ = self.model(X)
        loss = self.loss(Y_, Y)
        for key in self.train_metric.keys():
            self.train_metric[key].update(Y_, Y)

        self.log(
            "tr_output_mean", Y_.mean(), sync_dist=True, prog_bar=True, on_epoch=True
        )
        self.log("train_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True)
        return loss

    def non_train_step(self, batch, name="no_name"):
        with torch.no_grad():
            X, Y = batch
            Y_ = self.model(X)
            for key in self.val_metric.keys():
                self.val_metric[key].update(Y_, Y)
            self.log(
                name + "output_mean",
                Y_.mean(),
                sync_dist=False,
                prog_bar=True,
                on_epoch=True,
            )

    def validation_step(self, batch, idx):
        self.non_train_step(batch, name="val_")

    def forward(self, batch):
        X, Y = batch
        Y_ = self.model(X)
        return Y_

    def test_step(self, batch, _):
        self.non_train_step(batch, name="test_")

    def on_train_epoch_end(self):
        for key in self.train_metric.keys():
            self.log(
                "train_" + key,
                self.train_metric[key].compute(),
                sync_dist=False,
                prog_bar=True,
                on_epoch=True,
            )
            self.train_metric[key].reset()

    def on_validation_epoch_end(self):
        for key in self.val_metric.keys():
            self.log(
                "val_" + key,
                self.val_metric[key].compute(),
                sync_dist=False,
                prog_bar=True,
                on_epoch=True,
            )
            self.val_metric[key].reset()

    def configure_optimizers(self):
        optim = opt.AdamW(
            self.model.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay,
        )
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='max',  # maximizing negative SHD (lower SHD is better)
            factor=0.5,  # less aggressive reduction
            patience=5,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": schedule,
                "monitor": "val_NegSHD",
                "interval": "epoch",
                "frequency": 1,
            }
        }
