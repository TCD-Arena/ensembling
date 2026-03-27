import hydra
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
import os
from omegaconf import OmegaConf


@hydra.main(
    version_base="1.3", config_path="config", config_name="2_train_ensembles.yaml"
)
def main(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    print("Loading")
    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    #logger: TensorBoardLogger = hydra.utils.instantiate(cfg.tensorboard)
    #logger.log_hyperparams(cfg)
    print("Loading done.")

    logger = CSVLogger(save_dir=cfg.directory, name=None)
    print("Logger initialized.")
    
    data_module.setup(0)
    print("Data module done.")

    # callbacks
    print("Monitoring metric:", cfg.monitor)

    checkpoint_callback = ModelCheckpoint(
        mode="max",
        dirpath=cfg.directory,
        save_top_k=1,
        monitor=cfg.monitor,
        save_last=True,
    )
    es = hydra.utils.instantiate(cfg.early_stopping, monitor=cfg.monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # save config of the run as yaml into the cfg.directory:
    # make directoy if it does not exist:
    if not os.path.exists(cfg.directory):
        os.makedirs(cfg.directory)
    with open(os.path.join(cfg.directory, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[
            es,
            lr_monitor,
            checkpoint_callback,
            # RichModelSummary(),
            RichProgressBar(),
        ],
    )

    trainer.fit(model=model, datamodule=data_module)


    print("Done. Check csv logs.")


if __name__ == "__main__":
    main()
