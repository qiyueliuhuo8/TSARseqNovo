from typing import Optional, Tuple
import h5py

import depthcharge
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import Dataset


class MSDataset(Dataset):
    def __init__(
        self,
        data_paths: list,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self._handle = None
        self.data_paths = data_paths

        self._length = []
        for data_path in data_paths:
            with h5py.File(data_path, "a") as index:
                # print(f"h5py file : {data_path}")
                if len(self._length) == 0:
                    self._length.append(index.attrs["n_spectra"])
                else:
                    self._length.append(index.attrs["n_spectra"]+self._length[-1])

    def get_row_col(self, idx):
        row = 0
        col = 0
        for i, length in enumerate(self._length):
            if idx < length:
                row = i
                if i == 0:
                    col = idx
                else:
                    col = idx - self._length[i-1]
                break
        return row, col

    def get_spectrum(self, idx):
        row, col = self.get_row_col(idx=idx)
        self._handle = h5py.File(self.data_paths[row], "r", rdcc_nbytes=int(3e8), rdcc_nslots=1024000,)
        idx = col

        metadata = self._handle["metadata"]
        spectra = self._handle["spectra"]
        annotations = self._handle["annotations"]
        features = self._handle["feature"]

        if idx == self._handle.attrs["n_spectra"]-1:
            start_offset = metadata["offset"][-1]
            end_offset = spectra.shape[0]
        else:
            start_offset = metadata["offset"][idx]
            end_offset = metadata["offset"][idx+1]

        spectrum = spectra[start_offset:end_offset]
        precursor = metadata[idx]
        sequence = annotations[idx].decode()
        feature = features[start_offset:end_offset]
        feature = torch.tensor(feature).float()

        ms = torch.tensor(np.vstack((spectrum["mz_array"], spectrum["intensity_array"]))).T.float()
        out = (
            ms,
            precursor["precursor_mz"],
            precursor["precursor_charge"],
            sequence,
            feature
        )
        return out
    
    def __getitem__(self, idx):
        return self.get_spectrum(idx)
    
    def __len__(self):
        return self._length[-1]
