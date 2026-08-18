import os
import pickle
import blosc2
import numpy as np
from torch.utils.data import Dataset


def load_patient_dirs(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class WBMRI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'dir'
        transform: optional transform applied to b2nd data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_dirs = load_patient_dirs(
            os.path.join(root_dir, "patient_dirs.pkl")
        )
        print(f"Total patients found: {len(self.patient_dirs)}")

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

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        pkl_file = "metadata.pkl"
        mask_file = "mask.b2nd"

        fat_path = os.path.join(patient_dir, "fat.b2nd")
        wat_path = os.path.join(patient_dir, "wat.b2nd")
        pkl_path = os.path.join(patient_dir, pkl_file)
        mask_path = os.path.join(patient_dir, mask_file)

        fat_data = self._load_b2nd(fat_path)
        wat_data = self._load_b2nd(wat_path)
        b2nd_data = np.stack([fat_data, wat_data], axis=0)

        mask_data = self._load_b2nd(mask_path)[None]
        inputs = {"img": b2nd_data, "mask": mask_data}

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        if self.transform:
            inputs = self.transform(inputs)

        return {
            "data": inputs["img"],
            "body_mask": inputs["mask"],
            "metadata": pkl_data,
            "patient": patient_dir,
            "modality": "both",
        }
