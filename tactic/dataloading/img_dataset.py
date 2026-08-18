import os
import blosc2
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


def load_patient_dirs(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class WBMRI_train_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        csv_path="/path/to/csv_files/dir/",
        instance=2,
        split="balanced train",
        transform=None,
        label_key="Breast cancer",
        disease="breast_cancer",
    ):
        """
        root_dir: path to 'dir'
        transform: optional transform applied to b2nd data
        """
        self.root_dir = root_dir
        self.transform = transform

        # get list of patients with an image
        patient_dirs = load_patient_dirs(os.path.join(root_dir, "patient_dirs.pkl"))
        imaging_eids = [
            int(n.split("/")[-1][:-2]) for n in patient_dirs if n.endswith("_2")
        ]

        # get list of patients in the tabular splits
        labels = pd.read_csv(os.path.join(csv_path, f"{disease}-{instance}.csv"))
        split_csv = pd.read_csv(
            os.path.join(csv_path, f"splits-{disease}-{instance}.csv")
        )

        assert labels["eid"].equals(
            split_csv["eid"]
        ), "EIDs don't match (labels vs split)"

        eids_in_split = set(split_csv.loc[split_csv[split] == 1, "eid"])
        common_eids = list(eids_in_split & set(imaging_eids))
        split_csv = split_csv[split_csv["eid"].isin(common_eids)]

        self.patient_dirs = [
            p
            for p in patient_dirs
            if (int(p.split("/")[-1][:-2]) in common_eids and p.endswith("_2"))
        ]

        labels = labels.join(split_csv[split])
        labels = labels[labels[split].notna()]
        labels = labels.reset_index(drop=True)

        self.labels = labels
        self.label_key = label_key
        print(f"Total patients found: {len(self.labels)} for disease {disease}")

    def __len__(self):
        return len(self.patient_dirs)

    def _load_b2nd(self, path):
        """
        Replace this with your actual .b2nd loading logic
        """
        with open(path, "rb") as f:
            data = f.read()

        data = blosc2.unpack_array(data)
        return data

    def __getitem__(self, index):
        ## Load the images
        patient_dir = self.patient_dirs[index]

        fat_path = os.path.join(patient_dir, "fat.b2nd")
        fat_data = self._load_b2nd(fat_path)
        wat_path = os.path.join(patient_dir, "wat.b2nd")
        wat_data = self._load_b2nd(wat_path)

        mask_path = os.path.join(patient_dir, "mask.b2nd")
        body_mask = self._load_b2nd(mask_path)

        eid = int(patient_dir.split("/")[-1][:-2])

        b2nd_data = np.stack([fat_data, wat_data], axis=0)
        b2nd_dict = {"img": b2nd_data, "mask": body_mask[None]}

        # do the transformations
        if self.transform:
            b2nd_dict = self.transform(b2nd_dict)
            b2nd_data = b2nd_dict["img"]
            body_mask = b2nd_dict["mask"]

        # load image metadata
        pkl_path = os.path.join(patient_dir, "metadata.pkl")
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        ## Load the annotations
        label_row = self.labels.loc[self.labels["eid"] == eid, ["eid", self.label_key]]
        return {
            "data": b2nd_data,
            "body_mask": body_mask,
            "label": torch.tensor(
                label_row[self.label_key].iloc[0], dtype=torch.float32
            ),
            "tab_eid": torch.tensor(label_row["eid"].iloc[0], dtype=torch.float32),
            "metadata": pkl_data,
            "patient": patient_dir,
            "modality": "both",
            # "modality": b2nd_file,
        }
