import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from cospred_model import pep_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MassSpecDataset(Dataset):
    """
    Handles all aspects of the data.
    """

    def __init__(self, filename):
        """
        initialize data and create torch tensors from numpy arrays

        Args:
            data_dir: (string) directory containing the dataset
            type: (string) train, test or val. Subdirectory containig splitted dataset

        """
        spectra = pep_utils.readmgf(filename)
        precursor = [pep_utils.get_precursor_charge_onehot(sp['charge']) for sp in spectra]
        ce = [sp['nce'] for sp in spectra]
        si = [pep_utils.get_sequence_integer(sp['pep']) for sp in spectra]
        seq = [sp['pep'] for sp in spectra]
        charge = [sp['charge'] for sp in spectra]
        self.x_precursor = torch.tensor(np.stack(charge), dtype=torch.long)
        self.x = torch.tensor(np.stack(si), dtype=torch.long)
        self.x_ce = torch.tensor(np.stack(ce))
        real_vectors = [pep_utils.spectrum2vectorn(sp['mz'], sp['it'], sp['mass'], pep_utils.BIN_SIZE,
                                                   sp['charge']) for sp in spectra]
        self.y = torch.tensor(np.stack(real_vectors), dtype=torch.float32)
        # loaded = torch.load(filename)
        # self.x = loaded['sequence_integer']
        # self.x_ce = loaded['collision_energy_aligned_normed']
        # self.x_precursor = loaded['precursor_charge_onehot']
        # self.y = loaded['intensities_raw']

    def __getitem__(self, index):
        return self.x[index], self.x_precursor[index], self.x_ce[index], self.y[index]

    def __len__(self):
        return len(self.x)