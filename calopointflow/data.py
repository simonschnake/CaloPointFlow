from typing import Callable, Optional
import pickle
import os
from urllib.error import URLError
import h5py
import numpy as np
import pytorch_lightning as pl

import torch
import torch.utils.data as data

from torchvision.datasets.utils import download_url, verify_str_arg, \
    check_integrity, calculate_md5

class CaloDataset(data.Dataset):
    directory = ""
    url = ""
    resources = []

    def __init__(
        self,
        root: str,
        split: str = "train",
        coords_transform: Optional[Callable] = None,
        vals_transform: Optional[Callable] = None,
        incident_energies_transform: Optional[Callable] = None,
        download: bool = False,
        preprocess: bool = False
    ) -> None:
        super().__init__()
        self.root = root 

        self.split = verify_str_arg(split.lower(), "split", ("train", "val", "test"))

        if download:
            self.download()
        if preprocess:
            self.prepare_data()

        self._load_data()

        self.coords_transform = coords_transform
        self.vals_transform = vals_transform
        self.incident_energies_transform = incident_energies_transform

    def __getitem__(self, index):
        nnz = self.nnz[index]
        start = self.start_index[index]
        end = start + nnz

        coords = self.coords[start:end]
        vals = self.vals[start:end]
        incident_energies = self.incident_energies[index]

        if self.coords_transform is not None:
            coords = self.coords_transform(coords)
        if self.vals_transform is not None:
            vals = self.vals_transform(vals)
        if self.incident_energies_transform is not None:
            incident_energies = self.incident_energies_transform(incident_energies)

        return coords, vals, nnz, incident_energies
    
    def __len__(self):
        return len(self.incident_energies)
    
    def _load_data(self):
        data = np.load(os.path.join(self.root, self.directory, f"{self.split}.npz"))

        self.coords = torch.from_numpy(data["coords"])
        self.vals = torch.from_numpy(data["vals"])
        self.nnz = torch.from_numpy(data["nnz"])
        self.start_index = torch.from_numpy(data["start_index"])
        self.incident_energies = torch.from_numpy(data["incident_energies"])
    
    def _check_raw_files_integrity(self) -> bool:
        for fname, md5 in self.resources:
            if not self._check_integrity(fname, md5):
                return False
        return True

    def _check_preprocessed_files_integrity(self) -> bool:
        signatures_path = os.path.join(self.root, self.directory, "signatures.pkl")

        if not os.path.isfile(signatures_path):
            return False

        with open(signatures_path, "rb") as f:
            signatures = pickle.load(f)

        for fname, md5 in signatures.items():
            if not self._check_integrity(fname, md5):
                return False
        return True
    
    def _check_integrity(self, fname, md5) -> bool:
        f = os.path.join(self.root, self.directory, fname)
        return check_integrity(f, md5)

    def download(self) -> None:
        """Download the calo data if it doesn't exist already."""

        if self._check_raw_files_integrity():
            return

        os.makedirs(os.path.join(self.root, self.directory), exist_ok=True)

        # download files
        for fname, md5 in self.resources:
            try:
                print(f"Downloading {self.url}{fname}")
                download_url(
                    self.url + fname, 
                    root=os.path.join(self.root, self.directory),
                    filename=fname, 
                    md5=md5)
            except URLError:
                raise RuntimeError(f"Error downloading {fname}")
        
    def prepare_data(self) -> None:
        if self._check_preprocessed_files_integrity():
            return
        signature_path = os.path.join(self.root, self.directory, "signatures.pkl")
        with open(signature_path, "wb") as f:
            signatures = {split : "md5 wrong" for split in self.data_split}
            pickle.dump(signatures, f)

        for split, data_parts in self.data_split.items():
            self.prepare(split, data_parts)
    
    def prepare(self, split, data_parts):
        coords_list = []
        vals_list = []
        nnz_list = []
        incident_energies_list = []
        for d in data_parts:
            if type(d) == tuple:
                id, start, end = d
            else:
                id = d
                start = 0.
                end = 1.
            fname = self.resources[id][0]
            with h5py.File(os.path.join(self.root, self.directory, fname), "r") as f:
                if start == 0. and end == 1.:
                    showers = f["showers"][:]
                    incident_energies = f["incident_energies"][:]
                else:
                    start = int(start * f["showers"].shape[0])
                    end = int(end * f["showers"].shape[0])
                    showers = f["showers"][start:end]
                    incident_energies = f["incident_energies"][start:end]
                
            coords, vals, nnz, idx = self.to_point_clouds(showers, incident_energies)    
            coords_list.append(coords)
            vals_list.append(vals)
            nnz_list.append(nnz)
            incident_energies = incident_energies[idx].astype(np.float32) 
            incident_energies_list.append(incident_energies)

        coords = np.concatenate(coords_list)
        vals = np.concatenate(vals_list)
        nnz = np.concatenate(nnz_list)
        incident_energies = np.concatenate(incident_energies_list)

        start_index = np.zeros(nnz.shape, dtype=np.int64) # create start_index array
        start_index[1:] = np.cumsum(nnz)[:-1] # calculate start_index

        split_path = os.path.join(self.root, self.directory, f"{split}.npz")

        np.savez(
            split_path,
            coords=coords,
            vals=vals,
            nnz=nnz,
            start_index=start_index,
            incident_energies=incident_energies
        )

        signatures_path = os.path.join(self.root, self.directory, "signatures.pkl")

        with open(signatures_path, "rb") as f:
            signatures = pickle.load(f)
        with open(signatures_path, "wb") as f:
            signatures[split] = calculate_md5(split_path)
            pickle.dump(signatures, f)
    
    def to_point_clouds(
            self, 
            showers: np.ndarray, 
            incident_energies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.argwhere(showers > 0.)
        vals = showers[coords[:, 0], coords[:, 1]]
        idx, nnz = np.unique(coords[:, 0], return_counts=True)
        idx = np.sort(idx)
        coords = coords[:, 1:]
        vals = vals.astype(np.float32)

        return coords, vals, nnz, idx

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CaloDataset1Photons(CaloDataset):
    directory = "dataset_1_photons"
    url = "https://zenodo.org/record/8099322/files/"
    resources = [
        ("dataset_1_photons_1.hdf5", "005d2adeda7db034b388112661265656"),
        ("dataset_1_photons_2.hdf5", "4767715ed56e99565fd9c67340661e70"),
    ]

    data_split = {
        # split : resource_number, start_percentage, end_percentage
        "train": [(0, 0.0, 0.95)],
        "val": [(0, 0.95, 1.0)],
        "test": [1],
    }

class CaloDataset1Pions(CaloDataset):
    directory = "dataset_1_pions"
    url = "https://zenodo.org/record/8099322/files/"
    resources = [
        ("dataset_1_pions_1.hdf5", "6a5f52722064a1bcd8a0bc002f16515d"),
        ("dataset_1_pions_2.hdf5", "fee7457b40127bc23c8ab909e2638ca0"),
    ]

    data_split = {
        # split : resource_number, start_percentage, end_percentage
        "train": [(0, 0.0, 0.95)],
        "val": [(0, 0.95, 1.0)],
        "test": [1],
    }


class CaloDataset2(CaloDataset):
    directory = "dataset_2"
    url = "https://zenodo.org/record/6366271/files/"
    resources = [
        ("dataset_2_1.hdf5", "e590333e9a2da51b258288d74bd8357a"),
        ("dataset_2_2.hdf5", "7a56fd68aa53ded37c2ac445694d9736"),
    ]   

    data_split = {
        # split : resource_number, start_percentage, end_percentage
        "train": [(0, 0.0, 0.95)], 
        "val": [(0, 0.95, 1.0)], # 5% of the training data is used for validation
        "test": [1],
    }


class CaloDataset3(CaloDataset):
    directory = "dataset_3"
    url = "https://zenodo.org/record/6366324/files/"
    resources = [
        ("dataset_3_1.hdf5", "d33f03418aa311b0965452fcbef1ff87"),
        ("dataset_3_2.hdf5", "911dec2c40aafa64a07b7450b80881bf"),
        ("dataset_3_3.hdf5", "5ea62f6831a1821bb5035fafc9f030e8"),
        ("dataset_3_4.hdf5", "5e8d82e6a6c28c6914a3b602a1cfa299"),
    ]

    data_split = {
        # split : resource_number, start_percentage, end_percentage
        "train": [0, (1, 0.0, 0.9)],
        "val": [(1, 0.9, 1.0)],
        "test": [2, 3],
    }


class CaloDataModule(pl.LightningDataModule):
    calo_dataset = None
    coords_transform = None
    vals_transform = None
    incident_energies_transform = None

    def __init__(
            self, 
            data_dir,
            batch_size: int = 32,
            num_workers: int = 1
            ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def download_and_prepare_data(self):
        # Download the data if necessary
        self.calo_dataset(self.data_dir, download=True, preprocess=True)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.calo_dataset(
                self.data_dir, 
                split="train",
                coords_transform=self.coords_transform,
                vals_transform=self.vals_transform,
                incident_energies_transform=self.incident_energies_transform)
    
            self.val_dataset = self.calo_dataset(
                self.data_dir,
                split="val",
                coords_transform=self.coords_transform,
                vals_transform=self.vals_transform,
                incident_energies_transform=self.incident_energies_transform)

        if stage == "val":
            self.val_dataset = self.calo_dataset(
                self.data_dir,
                split="val",
                coords_transform=self.coords_transform,
                vals_transform=self.vals_transform,
                incident_energies_transform=self.incident_energies_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test_dataset = self.calo_dataset(
                self.data_dir,
                split="test",
                coords_transform=self.coords_transform,
                vals_transform=self.vals_transform,
                incident_energies_transform=self.incident_energies_transform)
 
        return super().setup(stage)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )
    
    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )
    
    def collate_fn(self, batch):
        coords, vals, nnz, incident_energies = zip(*batch)

        coords = torch.cat(coords)
        vals = torch.cat(vals)
        nnz = torch.stack(nnz)
        incident_energies = torch.stack(incident_energies)

        return coords, vals, nnz, incident_energies


class CaloDataModule1Photons(CaloDataModule):
    calo_dataset = CaloDataset1Photons

class CaloDataModule1Pions(CaloDataModule):
    calo_dataset = CaloDataset1Pions

class CaloDataModule2(CaloDataModule):
    calo_dataset = CaloDataset2

    def shower_transform(self, x):
        x = super().shower_transform(x)
        return x.view(-1, 45, 16, 9)

class CaloDataModule3(CaloDataModule):
    calo_dataset = CaloDataset3

if __name__ == '__main__':
    # Path for the datasets 
    path = "/beegfs/desy/user/schnakes/calochallenge_data"

    for d in [
        CaloDataModule1Photons, CaloDataModule1Pions,
        CaloDataModule2, CaloDataModule3
        ]: 
        d = d(path)
        d.download_and_prepare_data()
        print(d)

