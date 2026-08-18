import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from jup_builders import build_model, get_dataloaders
from jup_models import self_attention_rollout


def get_points(attr_dict, num_points, device):
    attributes, click_mask = attr_dict["attributes"]
    edge_attr = attr_dict["edge_attr"]
    mask = attr_dict["attn_mask"]

    attr_input = attributes.to(device)
    edge_attr = edge_attr.to(device)
    mask = mask.to(device)

    return attr_input, edge_attr, mask


parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="breast_cancer")
parser.add_argument("--task_name", type=str, default="Breast_imTab_load")
parser.add_argument("--pretrained_tab", type=str, default="")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--merge_tokens", type=int, nargs="+", default=[4, 4, 2])
parser.add_argument("--img_size", type=int, nargs="+", default=[224, 160, 352])
parser.add_argument("--patch_size", type=int, nargs="+", default=[8, 8, 8])
args = parser.parse_args()

dataloader = get_dataloaders(args)

num_names = dataloader.collate_fn.data_prep_module.num_col_names_
cat_names = dataloader.collate_fn.data_prep_module.cat_col_names_
tarte_names = (
    dataloader.collate_fn.data_prep_module.col_names_
)  # output_order [cat, num]
input_names = list(dataloader.dataset.tabular_data.columns)  # input order

model = build_model(args)
model.eval()

print("done")

dicts = {"attn": [], "mask": [], "sample": []}
classes = np.arange(1, args.num_classes)

attr_keys = ["attributes", "edge_attr", "attn_mask"]
count = 0
for sample in tqdm(dataloader):
    count += 1
    image = sample["data"].float().to(args.device)
    body_mask = sample["body_mask"].float().to(args.device)
    labels = sample["label"].type(torch.long)  # B

    attributes = {k: sample[k] for k in attr_keys}
    test_attributes = attributes["attributes"][1].shape[1]

    # Tabular
    attr_input = get_points(attributes.copy(), test_attributes, args.device)
    # sparse_embeddings = model.tabular_encoder(*attr_input)

    # Get TARTE attn
    model.tabular_encoder(*attr_input)
    attn_matrices = []
    for j in range(3):
        mat = model.tabular_encoder.tabular_encoder.transformer_encoder.layers[
            j
        ].saved_attn
        attn_matrices.append(mat.unsqueeze(1).cpu())
    combined = self_attention_rollout(attn_matrices, "cpu")
    dicts["attn"].append(combined[:, 0, 1:])

    tmp_mask = attributes["attributes"][1] == 1
    dicts["mask"].append(tmp_mask.cpu())

    dicts["sample"].extend(sample["per_sample"])


merged = torch.cat(dicts["attn"]).cpu()

accumulation = torch.zeros(len(input_names))
counts = torch.zeros(len(input_names))
for i in range(len(merged)):
    new_idxs = [input_names.index(j) for j in dicts["sample"][i]]
    accumulation[new_idxs] += merged[i, : len(new_idxs)]
    counts[new_idxs] += 1

means = accumulation / counts
means = torch.nan_to_num(means, nan=-torch.inf, posinf=-torch.inf)

probabilities = torch.softmax(means * 100, dim=0)

ordered, order = probabilities.sort(descending=True)

ordered_names = [input_names[i] for i in order]
feature_importance = {
    "importance": ordered.tolist(),
    "attributes": ordered_names,
}
with open(f"results_tab/{args.disease}_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=4)

n_features = 10


def wrap_labels(labels, width=30):
    return ["\n".join(textwrap.wrap(l, width)) for l in labels]


importance = feature_importance["importance"]
ordered_names = feature_importance["attributes"]
features_wrapped = wrap_labels(ordered_names)

# Plot
plt.figure(figsize=(12, 8))
bars = plt.bar(
    features_wrapped[:n_features],
    importance[:n_features],
    color="#27476E",
    edgecolor="none",
)

# Labels and title
plt.ylabel("Attention Score", fontsize=14)
plt.title(
    f"Top {n_features} attention scores per attribute for {args.disease} classification",
    fontsize=16,
    pad=20,
)

# Rotate x labels slightly and reduce font size
plt.xticks(rotation=30, ha="right", fontsize=12)

# Add some space below and tighten layout
# plt.subplots_adjust(bottom=0.3)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save or show
plt.savefig(
    f"results_tab/top{n_features}_feature_importance_{args.disease}.pdf",
    bbox_inches="tight",
)
# plt.show()


mask_path = "/lustre/groups/iml/projects/marta/cardiac_representations/random_masks"
path_root = "most_important_test_"
cur_mask = path_root + "10.npy"
mask_data = np.load(os.path.join(mask_path, cur_mask), allow_pickle=True).item()
eids = mask_data["eid"]
mask = mask_data["mask"]
columns = mask_data["columns"]
