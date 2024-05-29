import logging
import os
import h5py
import sys
import uuid
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Union
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
import depthcharge

from ..data.dataloaders import MSDatasetDataModule
from .model_interface import MInterface
from .. import evaluate 

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

logger = logging.getLogger("TSARseqNovo")

def train_MSDataset(
    train_data_path: list,
    valid_data_path: list,
    model_name: str,
    config: Dict[str, Any],
) -> None:
    dataloader_params = dict(
        batch_size=config["hparameters"]["batch_size"],
        n_workers=config["dataloader"]["n_workers"],
    )
    train_loader = MSDatasetDataModule(
        train_data_path=train_data_path, **dataloader_params
    )
    train_loader.setup("fit")
    print(f"training set lengths :{train_loader.train_dataset._length}\n")

    valid_loader = MSDatasetDataModule(
        valid_data_path=valid_data_path, **dataloader_params
    )
    valid_loader.setup("validate")
    print(f"valid set lengths :{valid_loader.valid_dataset._length}\n")

    model_name = config['model']['model_name']
    model_parameters = {
        "dim_model": config["model"]["transformer_width"],
        "n_layers": config["model"]["transformer_layers"],
        "dim_aa_embedding": config["model"]["dim_aa_embedding"],
        "k_step": config["model"]["k_step"],
        "max_charge": config["model"]["max_charge"],
        "n_heads": config["model"]["transformer_heads"],
        "batch_first": config["model"]["batch_first"],
        "dropout": config["hparameters"]["dropout"],
        "max_out_len": config["model"]["seq_len"],
        "batch_size": config["hparameters"]["batch_size"],
        "lr_scheduler": config["hparameters"]["lr_scheduler"],
        "warmup_iters": config["hparameters"]["warmup_iters"],
        "max_iters": config["hparameters"]["max_iters"],
        "weight_decay": config["hparameters"]["weight_decay"],
        "t_mult": config["hparameters"]["t_mult"],
        "lr_min": config["hparameters"]["lr_min"],
        "train_label_smoothing": config["hparameters"]["train_label_smmoothing"]
    }
    lr = config['hparameters']['lr']

    if config["model"]["resume"] is not None:
        minterface = MInterface.load_from_checkpoint(config["model"]["resume"], map_location={'cuda:0':'cuda:1'}, **model_parameters)
        resume_pth = config["model"]["resume"]
        print(f"resume path : {resume_pth}")
    else:
        minterface = MInterface(model_name, lr, **model_parameters)

    save_path = os.path.join(config['save_path'], config['experiment_name'])
    callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=save_path,
                save_top_k=10,
                monitor='valid_loss',
                mode='min',
                filename="{epoch:02d}-{valid_loss:.4f}",
            )
        ]
    
    gpus = config['Trainer']['device'].split(',')
    gpus = [int(f) for f in gpus]
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        devices=gpus,
        enable_checkpointing=True,
        logger=TensorBoardLogger(save_dir=save_path, version=1, name="lightning_logs"),
        max_epochs=config['Trainer']["max_epochs"],
        num_sanity_val_steps=0,
        strategy=_get_strategy(),
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    
    # Train the model.
    trainer.fit(
        minterface, train_loader.train_dataloader(), valid_loader.val_dataloader()
    )

def predict_MSDataset(    
    valid_data_path: list,
    config: Dict[str, Any],
    ):
    dataloader_params = dict(
        batch_size=config["hparameters"]["batch_size"],
        n_workers=config["dataloader"]["n_workers"],
    )

    valid_loader = MSDatasetDataModule(
        valid_data_path=valid_data_path, **dataloader_params
    )
    valid_loader.setup("validate")
    print(f"valid set lengths :{valid_loader.valid_dataset._length}\n")
    gpus = config['Trainer']['device'].split(',')
    gpus = [int(f) for f in gpus]

    if config["model"]["resume"] is not None:
        ckpt = torch.load(config["model"]["resume"], map_location={'cuda:0':'cuda:'+str(gpus[0])})
        hyper_parameters = ckpt["hyper_parameters"]
        print(hyper_parameters)
        minterface = MInterface( **hyper_parameters)
        minterface.load_state_dict(ckpt["state_dict"])
        minterface.max_out_len = config["dataset"]["max_out_len"]
    save_path = os.path.join(config['save_path'], config['experiment_name'])
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        enable_checkpointing=True,
        logger=TensorBoardLogger(save_dir=save_path, version=1, name="lightning_logs"),
        num_sanity_val_steps=0,
        strategy=_get_strategy(),
    )

    predictions = trainer.predict(
        minterface, valid_loader.val_dataloader(), 
    )
    predictions = minterface.test_predictions

    sequence = []
    sequence_pred = []
    eqs = []
    for batch_prediction in  predictions:
        pred, anno, mat = batch_prediction
        for i in range(len(anno)):
            sequence.append(anno[i])
            sequence_pred.append(pred[i])
            eqs.append(mat[i])
    df = pd.DataFrame()
    df["sequence"] = sequence
    df["sequence_pred"] = sequence_pred
    df["eqs"] = eqs
    df.to_csv(os.path.join(save_path, "psms.csv"))

    aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                sequence_pred,
                sequence,
                minterface.model.vocabulary,
            )
        )
    print(f"aa recall :{aa_recall}")
    print(f"aa precision: {aa_precision}")
    print(f"peptide precision: {pep_precision}")

def _get_strategy() -> Optional[DDPStrategy]:

    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return None