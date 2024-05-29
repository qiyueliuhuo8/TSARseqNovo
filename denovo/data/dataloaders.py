import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from .datasets import MSDataset

class MSDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: list = None,
        valid_data_path: list = None,
        test_data_path: list = None,
        batch_size: int = 128,
        n_workers: Optional[int] = 15,
    ):
        super().__init__()
        self.train_path = train_data_path
        self.valid_path = valid_data_path
        self.test_path = test_data_path
        self.b_size = batch_size
        self.n_workers = n_workers
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: str = None,) -> None:
        if stage in [None, "fit", "validate"]:
            make_dataset = functools.partial(
                MSDataset,
            )
            if self.train_path is not None:
                self.train_dataset = make_dataset(self.train_path)
            if self.valid_path is not None:
                self.valid_dataset = make_dataset(self.valid_path)
        if stage in [None, "test"]:
            make_dataset = functools.partial(
                MSDataset,
            )
            self.test_dataset = make_dataset(self.test_path)

    def _make_loader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.b_size,
            collate_fn=prepare_batch_MSDataset,
            pin_memory=True,
            num_workers=self.n_workers,
        )
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)
    
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset)
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset)

def prepare_batch_MSDataset(
    batch: List[Tuple[torch.Tensor, float, int, str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    spectra, precursor_mzs, precursor_charges, spectrum_ids, features = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return spectra, precursors, np.asarray(spectrum_ids), features
    # return batch, spectra, precursors, np.asarray(spectrum_ids), features