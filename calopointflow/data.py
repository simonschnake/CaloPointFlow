import os
import pickle
from urllib.error import URLError

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torchvision.datasets.utils import (
    calculate_md5,
    check_integrity,
    download_url,
    verify_str_arg,
)


class CaloDataset(data.Dataset):
    type = "voxel"

    def __init__(
        self,
        directory: str,
        url: str,
        resources: list[tuple[str, str]],
        data_split: dict[str, list[tuple[int, float, float]]],
        root: str,
        split: str = "train",
        download: bool = False,
        preprocess: bool = False,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.url = url
        self.resources = resources
        self.data_split = data_split
        self.root = root

        self.split = verify_str_arg(
            split.lower(), "split", ("train", "val", "test", "debug")
        )

        self.skip_integrity_check = True if split == "debug" else False

        if download:
            self.download()
        if preprocess:
            self.prepare_data()

        self._load_data()

    def __getitem__(self, index):
        shower = torch.from_numpy(self.showers[index])
        incident_energies = torch.from_numpy(self.incident_energies[index])
        return shower, incident_energies

    def __len__(self):
        return len(self.incident_energies)

    def _load_data(self):
        data = np.load(
            os.path.join(self.root, self.directory, f"{self.type}_{self.split}.npz")
        )
        self.showers = data["showers"]
        self.incident_energies = data["incident_energies"]

    def _check_raw_files_integrity(self) -> bool:
        for fname, md5 in self.resources:
            if not self._check_integrity(fname, md5):
                return False
        return True

    def _check_exists(self) -> bool:
        if self.skip_integrity_check:
            return True
        for fname, md5 in self.resources:
            if not self._check_integrity(fname, md5):
                return False
        return True

    def _check_preprocessed_files_integrity(self) -> bool:
        if self.skip_integrity_check:
            return True
        signatures_path = os.path.join(
            self.root, self.directory, f"{self.type}_signatures.pkl"
        )

        if not os.path.isfile(signatures_path):
            return False

        with open(signatures_path, "rb") as f:
            try:
                signatures = pickle.load(f)
            except EOFError:
                return False

        for split, md5 in signatures.items():
            if not self._check_integrity(f"{self.type}_{split}.npz", md5):
                return False
        return True

    def _check_integrity(self, fname, md5) -> bool:
        if self.skip_integrity_check:
            return True
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
                    md5=md5,
                )
            except URLError:
                raise RuntimeError(f"Error downloading {fname}")

    def prepare_data(self) -> None:
        if self._check_preprocessed_files_integrity():
            return
        signature_path = os.path.join(self.root, self.directory, f"{self.type}_signatures.pkl")
        with open(signature_path, "wb") as f:
            signatures = {split: "md5 wrong" for split in self.data_split}
            pickle.dump(signatures, f)

        for split, data_parts in self.data_split.items():
            self._prepare(split, data_parts)

    def _prepare(self, split, data_parts):
        showers_list = []
        incident_energies_list = []
        for d in data_parts:
            if type(d) == tuple:
                id, start, end = d
            else:
                id = d
                start = 0.0
                end = 1.0
            fname = self.resources[id][0]
            with h5py.File(os.path.join(self.root, self.directory, fname), "r") as f:
                if start == 0.0 and end == 1.0:
                    showers = f["showers"][:]
                    incident_energies = f["incident_energies"][:]
                else:
                    start = int(start * f["showers"].shape[0])
                    end = int(end * f["showers"].shape[0])
                    showers = f["showers"][start:end]
                    incident_energies = f["incident_energies"][start:end]

            showers_list.append(showers)
            incident_energies_list.append(incident_energies)

        showers = np.concatenate(showers_list)
        incident_energies = np.concatenate(incident_energies_list)

        split_path = os.path.join(self.root, self.directory, f"{self.type}_{split}.npz")

        np.savez(
            split_path,
            showers=showers.astype(np.float32),
            incident_energies=incident_energies.astype(np.float32),
        )

        signature_path = os.path.join(self.root, self.directory, f"{self.type}_signatures.pkl")

        with open(signature_path, "rb") as f:
            signatures = pickle.load(f)
        with open(signature_path, "wb") as f:
            signatures[split] = calculate_md5(split_path)
            pickle.dump(signatures, f)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class CaloPointCloudDataset(CaloDataset):
    type = "point"

    def __getitem__(self, index):
        nnz = self.nnz[index]
        start = self.start_index[index]
        end = start + nnz

        coords = self.coords[start:end]
        vals = self.vals[start:end]
        incident_energies = self.incident_energies[index]

        return coords, vals, nnz, incident_energies

    def _load_data(self):
        data = np.load(os.path.join(self.root, self.directory, f"{self.type}_{self.split}.npz"))

        self.coords = torch.from_numpy(data["coords"])
        self.vals = torch.from_numpy(data["vals"])
        self.nnz = torch.from_numpy(data["nnz"])
        self.start_index = torch.from_numpy(data["start_index"])
        self.incident_energies = torch.from_numpy(data["incident_energies"])

    def _prepare(self, split, data_parts):
        coords_list = []
        vals_list = []
        nnz_list = []
        incident_energies_list = []
        for d in data_parts:
            if type(d) is tuple:
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

        split_path = os.path.join(self.root, self.directory, f"{self.type}_{split}.npz")

        np.savez(
            split_path,
            coords=coords,
            vals=vals,
            nnz=nnz,
            start_index=start_index,
            incident_energies=incident_energies
        )

        signatures_path = os.path.join(self.root, self.directory, f"{self.type}_signatures.pkl")

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


class CaloDataModule(pl.LightningDataModule):
    directory: str = None
    url: str = None
    resources: list[tuple[str, str]] = None
    data_split: dict[str, list[tuple[int, float, float]]] = None

    def __init__(
        self,
        data_dir,
        batch_size: int = 32,
        num_workers: int = 1,
        download: bool = True,
        preprocess: bool = True,
        debug: bool = False,
        point_cloud: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.preprocess = preprocess
        self.debug = debug
        self.point_cloud = point_cloud

    def calo_dataset(self, **kwargs) -> CaloDataset:
        if self.point_cloud:
            return CaloPointCloudDataset(
                self.directory,
                self.url,
                self.resources,
                self.data_split,
                self.data_dir,
                **kwargs)
        else:
            return CaloDataset(
                self.directory,
                self.url,
                self.resources,
                self.data_split,
                self.data_dir,
                **kwargs)

    def prepare_data(self):
        # Download the data if necessary
        return self.calo_dataset(
            split = "debug" if self.debug else "train",
            download=self.download,
            preprocess=self.preprocess,
        )

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.calo_dataset(
                split="train" if not self.debug else "debug",
            )

            self.val_dataset = self.calo_dataset(
                split="val" if not self.debug else "debug",
            )

        if stage == "val":
            self.val_dataset = self.calo_dataset(
                split="val" if not self.debug else "debug",
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test_dataset = self.calo_dataset(
                split="test" if not self.debug else "debug",
            )

        return super().setup(stage)

    def _dataloader(self, **kwargs):
        return data.DataLoader(
            **kwargs,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.debug is False else 0,
            pin_memory=True,
            collate_fn=self.point_cloud_collate_fn if self.point_cloud else None,
        )

    def train_dataloader(self):
        return self._dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(dataset=self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(dataset=self.test_dataset, shuffle=False)

    def point_cloud_collate_fn(self, batch):
        coords, vals, nnz, incident_energies = zip(*batch)

        coords = torch.cat(coords)
        vals = torch.cat(vals)
        nnz = torch.stack(nnz)
        incident_energies = torch.stack(incident_energies)

        return coords, vals, nnz, incident_energies


class CaloDataModule1Photons(CaloDataModule):
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
        "debug": [(0, 0.0, 0.01)],  # 1% of the training data is used for debugging
    }

class CaloDataModule1Pions(CaloDataModule):
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


class CaloDataModule2(CaloDataModule):
    directory = "dataset_2"
    url = "https://zenodo.org/record/6366271/files/"
    resources = [
        ("dataset_2_1.hdf5", "e590333e9a2da51b258288d74bd8357a"),
        ("dataset_2_2.hdf5", "7a56fd68aa53ded37c2ac445694d9736"),
    ]

    data_split = {
        # split : resource_number, start_percentage, end_percentage
        "train": [(0, 0.0, 0.95)],
        "val": [(0, 0.95, 1.0)],  # 5% of the training data is used for validation
        "test": [1],
        "debug": [(0, 0.0, 0.01)],  # 1% of the training data is used for debugging
    }


class CaloDataModule3(CaloDataModule):
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
        "debug": [(0, 0.0, 0.01)],  # 1% of the training data is used for debugging
    }


if __name__ == "__main__":
    # Path for the datasets
    #path = "/beegfs/desy/user/schnakes/calochallenge_data"
    path = "/dev/shm/schnakes/datasets"

    for DataModule in [
        #CaloDataModule1Photons,
        #CaloDataModule1Pions,
        CaloDataModule2,
        #CaloDataModule3,
    ]:
        d = DataModule(path, point_cloud=True)
        d.prepare_data()
        print(d)

        d = DataModule(path, point_cloud=False)
        d.prepare_data()
        print(d)