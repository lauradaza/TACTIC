import os
import blosc2
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tactic.utils.attributes import ATTRIBUTES
from tactic.dataloading.img_dataset import load_patient_dirs

ATTRIBUTES = ATTRIBUTES


class WBImgTabDataset(Dataset):
    def __init__(
        self,
        root_dir,
        csv_path,
        instance="2",
        split="balanced train",
        simulate_missing=True,
        missing_tabular=False,
        selected_attr=[],
        transform=None,
        label_key="Breast cancer",
        disease="Breast cancer",
    ):

        # get list of patients with an image
        patient_dirs = load_patient_dirs(os.path.join(root_dir, "patient_dirs.pkl"))
        imaging_eids = [
            int(n.split("/")[-1][:-2]) for n in patient_dirs if n.endswith("_2")
        ]

        # Load CSV files
        tabular_data = pd.read_csv(os.path.join(csv_path, f"attributes-{instance}.csv"))
        tabular_data["eid"] = tabular_data["eid"].astype(int)

        labels = pd.read_csv(os.path.join(csv_path, f"{disease}-{instance}.csv"))
        split_csv = pd.read_csv(
            os.path.join(csv_path, f"splits-{disease}-{instance}.csv")
        )

        # Ensure that the eids match across all dataframes
        assert labels["eid"].equals(
            split_csv["eid"]
        ), "EIDs don't match (labels vs split)"
        assert tabular_data["eid"].equals(
            split_csv["eid"]
        ), "EIDs don't match (data vs split)"

        eids_in_split = set(split_csv.loc[split_csv[split] == 1, "eid"])
        common_eids = list(eids_in_split & set(imaging_eids))
        split_csv = split_csv[split_csv["eid"].isin(common_eids)]

        self.patient_dirs = [
            p
            for p in patient_dirs
            if (int(p.split("/")[-1][:-2]) in common_eids and p.endswith("_2"))
        ]

        tabular_data = tabular_data.join(split_csv[split])
        tabular_data = tabular_data[
            tabular_data[split].notna() & (tabular_data[split] != 0)
        ]
        tabular_data = tabular_data.drop(columns=[split])
        tabular_data = tabular_data.reset_index(drop=True)

        self.eid = tabular_data.pop("eid").to_list()

        labels = labels.join(split_csv[split])
        labels = labels[labels[split].notna() & (labels[split] != 0)]
        labels = labels.reset_index(drop=True)

        self.labels = labels

        # Store the filtered EIDs and CADs
        drop_cols = tabular_data.columns[tabular_data.nunique(dropna=True) <= 1]
        self.tabular_data = tabular_data.drop(columns=drop_cols)
        self.attributes = self.tabular_data.columns.tolist()

        self.simulate_missing = simulate_missing

        self.missing_tabular = missing_tabular
        self.selected_attr = selected_attr
        self.transform = transform
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

        ## Load tabular data
        # select values
        if len(self.selected_attr) != 0:
            chosen = self.selected_attr
        else:
            chosen = self.attributes

        # tabular_data = self.tabular_data.loc[self.labels["eid"] == eid, chosen]
        tab_eid = self.eid.index(eid)
        tabular_data = self.tabular_data.iloc[tab_eid]

        attributes = {}
        mask = []
        for col in chosen:
            val = tabular_data[col]
            if not isinstance(val, str):
                if not np.isnan(val):
                    attributes[col] = val
                    mask.append(1)
                else:
                    attributes[col] = np.nan
                    mask.append(0.001)
            else:
                mask.append(1)
        mask = torch.tensor(mask)

        ## Load the annotations
        label_row = self.labels.loc[self.labels["eid"] == eid, [self.label_key]]
        return {
            "images": b2nd_data,
            "body_mask": body_mask,
            "attributes": attributes,
            "mask": mask,
            "metadata": pkl_data,
            "eid": torch.tensor(eid),
            "label": torch.tensor(
                label_row[self.label_key].iloc[0], dtype=torch.float32
            ),
            "patient": patient_dir,
            "modality": "both",
        }


class WBImgTabCollator:
    def __init__(self, data_prep_module, order=None):
        self.data_prep_module = data_prep_module
        self.order = order

    def __call__(self, batch):
        combined_dicts = []
        for b in batch:
            combined_dicts.append(b["attributes"])

        df_combined = pd.DataFrame(combined_dicts)

        all_expected_cols = getattr(
            self.data_prep_module, "num_col_names_", []
        ) + getattr(self.data_prep_module, "cat_col_names_", [])

        missing_cols = [c for c in all_expected_cols if c not in df_combined.columns]

        if missing_cols:
            df_combined = pd.concat(
                [
                    df_combined,
                    pd.DataFrame(np.nan, index=df_combined.index, columns=missing_cols),
                ],
                axis=1,
            )

        # --- Preprocess once ---
        preprocessed = self.data_prep_module.transform(df_combined)
        idxs, xs, edge_attrs, masks, _ = zip(*preprocessed)
        xs, edge_attrs, masks = map(torch.stack, (xs, edge_attrs, masks))

        feature_order = self.data_prep_module.col_names_
        all_cat_cols = getattr(self.data_prep_module, "cat_col_names_", [])
        all_num_cols = getattr(self.data_prep_module, "num_col_names_", [])
        # Assuming dat_col_names_ holds any columns identified as datetime and treated as numerical
        all_dat_cols = getattr(self.data_prep_module, "dat_col_names_", [])
        per_sample_feature_order = []
        for index, row in df_combined.iterrows():
            # 1. Get the present (non-NaN) categorical features, in the global order
            present_cat = [
                col for col in all_cat_cols if col in row and not pd.isna(row[col])
            ]

            # 2. Get the present numerical features, in the global order
            present_num = [
                col for col in all_num_cols if col in row and not pd.isna(row[col])
            ]

            # 3. Get the present datetime features, in the global order
            present_dat = [
                col for col in all_dat_cols if col in row and not pd.isna(row[col])
            ]

            # 4. Stack them in the order used by TARTE_TablePreprocessor (Cat -> Num -> Dat)
            # This list is the precise mapping key for the active tokens j=1, j=2, ...
            stacked_order = present_cat + present_num + present_dat

            per_sample_feature_order.append(stacked_order)

        if self.order is not None:
            if isinstance(self.order, list):
                out_order = self.order
            for b in range(len(batch)):
                if isinstance(self.order, dict):
                    order_key = self.order["eid"].index[
                        self.order["eid"] == batch[b]["eid"].item()
                    ]
                    out_order = self.order["mask"][order_key.item()]
                    out_order = [x for x in out_order if x in all_expected_cols]

                idx_map = {
                    name: i for i, name in enumerate(per_sample_feature_order[b])
                }
                reorder_idx = [
                    idx_map[name] + 1 if name in idx_map else -1 for name in out_order
                ]  # +1 because 0 is [CLS]
                reorder_idx = [0] + reorder_idx
                xs[b] = xs[b][reorder_idx]
                edge_attrs[b] = edge_attrs[b][reorder_idx]
                masks[b] = masks[b][reorder_idx]
                per_sample_feature_order[b] = out_order

        return {
            "data": torch.stack([b["images"] for b in batch]),
            "body_mask": torch.stack([b["body_mask"] for b in batch]),
            "attributes": (
                xs,
                torch.stack([b["mask"] for b in batch]),
            ),
            "edge_attr": edge_attrs,  # edge features
            "attn_mask": masks,  # attention mask
            "eid": torch.stack([b["eid"] for b in batch]),  # duplicated eids
            "label": torch.stack([b["label"] for b in batch]),
            "feature_order": feature_order,
            "per_sample": per_sample_feature_order,
        }
