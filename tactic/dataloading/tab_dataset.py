import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tactic.utils.attributes import ATTRIBUTES

ATTRIBUTES = ATTRIBUTES


class MRClassificationDatasetH5(Dataset):
    def __init__(
        self,
        h5_root,
        instance="2",
        split="balanced train",
        simulate_missing=True,
        missing_tabular=False,
        selected_attr=[],
        label_key="Breast cancer",
        disease="breast_cancer",
    ):
        # Load CSV files
        self.tabular_data = pd.read_csv(
            os.path.join(h5_root, f"attributes-{instance}.csv")
        )
        self.tabular_data["eid"] = self.tabular_data["eid"].astype(int)

        labels = pd.read_csv(os.path.join(h5_root, f"{disease}-{instance}.csv"))
        self.split = pd.read_csv(
            os.path.join(h5_root, f"splits-{disease}-{instance}.csv")
        )

        # Ensure that the eids match across all dataframes
        assert labels["eid"].equals(
            self.split["eid"]
        ), "EIDs don't match (labels vs split)"
        assert self.tabular_data["eid"].equals(
            self.split["eid"]
        ), "EIDs don't match (data vs split)"

        self.split = self.split[split]

        self.tabular_data = self.tabular_data.join(self.split)
        self.tabular_data = self.tabular_data[
            self.tabular_data[split].notna() & (self.tabular_data[split] != 0)
        ]
        self.tabular_data = self.tabular_data.drop(columns=[split])
        self.tabular_data = self.tabular_data.reset_index(drop=True)

        labels = labels.join(self.split)
        labels = labels[labels[split].notna() & (labels[split] != 0)]
        labels = labels.reset_index(drop=True)

        self.labels = labels

        # Store the filtered EIDs and CADs
        self.eid = self.tabular_data.pop("eid").to_numpy()
        drop_cols = self.tabular_data.columns[
            self.tabular_data.nunique(dropna=True) <= 1
        ]
        self.tabular_data = self.tabular_data.drop(columns=drop_cols)
        self.attributes = self.tabular_data.columns.tolist()

        self.simulate_missing = simulate_missing

        self.missing_tabular = missing_tabular
        self.selected_attr = selected_attr
        self.label_key = label_key
        print(f"Total patients found: {len(self.labels)} for disease {disease}")

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, index):
        eid = self.eid[index]
        tabular_data = self.tabular_data.iloc[index]
        # select values
        if self.simulate_missing:
            # randomly sample between 20-100% of attributes
            chosen1 = random.sample(
                self.attributes,
                k=random.randint(1, int(0.2 * len(self.attributes))),
            )

            # randomly sample a subset from chosen1
            chosen2 = random.sample(chosen1, k=random.randint(0, len(chosen1)))
        else:
            # either choose the selected attributes or all
            if len(self.selected_attr) != 0:
                chosen1 = self.selected_attr
            else:
                chosen1 = self.attributes
            chosen2 = []

        attributes1 = {}
        for col in chosen1:
            val = tabular_data[col]
            if not isinstance(val, str):
                if not np.isnan(val):
                    attributes1[col] = val
                else:
                    attributes1[col] = np.nan
        attributes2 = {}
        for col in chosen2:
            val = tabular_data[col]
            if not isinstance(val, str):
                if not np.isnan(val):
                    attributes1[col] = val
                else:
                    attributes2[col] = np.nan

        label_row = self.labels.iloc[index]
        label = torch.tensor(label_row[self.label_key], dtype=torch.float32)

        tab_missing1 = int(len(chosen1) == 0)
        tab_missing2 = int(len(chosen2) == 0)
        return {
            "attributes1": attributes1,
            "attributes2": attributes2,
            "eid": eid,
            "label": label,
            "tab_missing1": torch.tensor(tab_missing1, dtype=torch.float32),
            "tab_missing2": torch.tensor(tab_missing2, dtype=torch.float32),
        }


class ClassificationCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        combined_dicts = []
        for b in batch:
            combined_dicts.append(b["attributes1"])
        for b in batch:
            combined_dicts.append(b["attributes2"])

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

        # --- Duplicate images and labels accordingly ---

        labels = torch.stack([b["label"] for b in batch])
        labels = torch.cat([labels, labels], dim=0)

        tab_missing1 = torch.stack([b["tab_missing1"] for b in batch])
        tab_missing2 = torch.stack([b["tab_missing2"] for b in batch])
        tab_missing = torch.cat([tab_missing1, tab_missing2], dim=0)
        eids = [b["eid"] for b in batch]
        eids = eids + eids  # duplicate list

        return {
            "x": xs,  # node features
            "edge_attr": edge_attrs,  # edge features
            "mask": masks,  # attention mask
            "eid": eids,  # duplicated eids
            "label": labels,
            "tab_missing": tab_missing,
        }


class RandomMaskDatasetH5(Dataset):
    def __init__(
        self,
        h5_root,
        masks_path,
        instance="2",
        split="test",
        full_tabular=False,
        importance=False,
        label_key="Cancer multi",
        disease="cancer",
    ):
        # Load CSV files instead
        self.tabular_data = pd.read_csv(
            os.path.join(h5_root, f"attributes-{instance}.csv")
        )
        self.tabular_data["eid"] = self.tabular_data["eid"].astype(int)

        labels = pd.read_csv(os.path.join(h5_root, f"{disease}-{instance}.csv"))
        self.split = pd.read_csv(
            os.path.join(h5_root, f"splits-{disease}-{instance}.csv")
        )

        # Ensure that the eids match across all dataframes
        assert labels["eid"].equals(
            self.split["eid"]
        ), "EIDs don't match (labels vs split)"
        assert self.tabular_data["eid"].equals(
            self.split["eid"]
        ), "EIDs don't match (data vs split)"

        self.split = self.split[split]

        self.tabular_data = self.tabular_data.join(self.split)
        self.tabular_data = self.tabular_data[
            self.tabular_data[split].notna() & (self.tabular_data[split] != 0)
        ]
        self.tabular_data = self.tabular_data.drop(columns=[split])
        self.tabular_data = self.tabular_data.reset_index(drop=True)

        labels = labels.join(self.split)
        labels = labels[labels[split].notna() & (labels[split] != 0)]
        labels = labels.reset_index(drop=True)

        self.labels = labels

        mask_data = np.load(
            masks_path.format(instance=instance), allow_pickle=True
        ).item()
        eids = mask_data["eid"]
        mask = mask_data["mask"]
        self.columns = mask_data["columns"]
        if importance:
            self.eid_to_mask = {eid: mask for _, eid in enumerate(eids)}
        else:
            self.eid_to_mask = {eid: mask[i] for i, eid in enumerate(eids)}

        # Store the filtered EIDs and CADs
        self.eid = self.tabular_data.pop("eid").to_numpy()
        drop_cols = self.tabular_data.columns[
            self.tabular_data.nunique(dropna=True) <= 1
        ]
        self.tabular_data = self.tabular_data.drop(columns=drop_cols)
        self.attributes = self.tabular_data.columns.tolist()

        drop_set = set(drop_cols)
        self.columns = [x for x in self.columns if x not in drop_set]

        self.full_tabular = full_tabular
        self.label_key = label_key

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, index):
        eid = self.eid[index]
        tabular_data = self.tabular_data.iloc[index]
        row_mask = self.eid_to_mask[eid]
        if self.full_tabular:
            cur_columns = self.attributes
        else:
            cur_columns = [col for col, keep in zip(self.columns, row_mask) if keep]

        attributes = {}
        for col in cur_columns:
            val = tabular_data[col]
            if not isinstance(val, str):
                if not np.isnan(val):
                    attributes[col] = val
                else:
                    attributes[col] = np.nan

        label_row = self.labels.iloc[index]
        label = torch.tensor(label_row[self.label_key], dtype=torch.float32)

        return {
            "attributes": attributes,
            "eid": eid,
            "label": label,
        }


class RandomMaskCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        # 1. Rebuild a DataFrame for tabular features
        combined_dicts = []
        for b in batch:
            combined_dicts.append(b["attributes"])

        df_batch = pd.DataFrame(combined_dicts)

        label = torch.stack([b["label"] for b in batch])

        all_expected_cols = getattr(
            self.data_prep_module, "num_col_names_", []
        ) + getattr(self.data_prep_module, "cat_col_names_", [])

        missing_cols = [c for c in all_expected_cols if c not in df_batch.columns]

        if missing_cols:
            df_batch = pd.concat(
                [
                    df_batch,
                    pd.DataFrame(np.nan, index=df_batch.index, columns=missing_cols),
                ],
                axis=1,
            )

        # for col in all_expected_cols:
        #     if col not in df_batch.columns:
        #         df_batch[col] = np.nan  # fill missing with NaN

        # 2. Run TARTE preprocessor
        preprocessed = self.data_prep_module.transform(df_batch)
        # preprocessed is a list of tuples: (idx, x, edge_attr, mask, y)

        # 3. Unpack
        idxs, xs, edge_attrs, masks, ys = zip(*preprocessed)

        # 4. Stack into tensors
        xs = torch.stack(xs)
        edge_attrs = torch.stack(edge_attrs)
        masks = torch.stack(masks)

        return {
            "x": xs,  # node features
            "edge_attr": edge_attrs,  # edge features
            "mask": masks,  # attention mask
            "eid": [b["eid"] for b in batch],
            "label": label,
        }
