import os
import sys
import click
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, F1Score

sys.path.insert(0, "/ictstr01/home/iml/marta.hasny/cardiac_representation/codes")
from tactic.modeling.tabular_encoder import TabularClassifier
from tactic.dataloading.tab_dataset import (
    RandomMaskCollator,
    RandomMaskDatasetH5,
    MRClassificationDatasetH5,
)

from tarte_ai import TARTE_TablePreprocessor

torch.cuda.empty_cache()

wandb.init(project="WB_tabular")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


@click.command()
@click.option(
    "--name",
    "-n",
    help="Where you want to store the models and results.",
    required=True,
    type=str,
)
@click.option(
    "--disease",
    "-d",
    required=False,
    default="cancer",
)
@click.option(
    "--order",
    "-o",
    required=False,
    default="rnd",
)
def main(name, disease, order):
    init_seeds()

    num_classes = 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    label_key = (
        disease.capitalize().replace("_", " ")
        if disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )

    train_data = MRClassificationDatasetH5(
        h5_root="/path/to/csv_files.dir/",
        instance="2",
        split="balanced train",
        simulate_missing=True,
        label_key=label_key,
        disease=disease,
    )

    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_prepper.fit(train_data.tabular_data)
    collator = RandomMaskCollator(tarte_tab_prepper)

    classifier = TabularClassifier(
        tabular_checkpoint="",
        num_classes=num_classes,
    )

    checkpoint = f"./results_tab/{name}/multilabel_cad/last.pth"
    state_dict = torch.load(checkpoint)
    classifier.load_state_dict(state_dict)

    classifier.to(device)
    classifier.eval()
    print("Model setup finished.", checkpoint)

    average = "macro"
    auc_metric = AUROC(task="binary")
    f1_metric = F1Score(task="binary")  # or "micro", "weighted", "none"

    tabular_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    mask_root = f"/path/to/csv_files.dir/test_{order}_masks"
    for percent in tabular_percent:
        if percent == 100:
            data = RandomMaskDatasetH5(
                h5_root="/path/to/csv_files/dir/",
                masks_path="/path/to/csv_files/dir/rnd_masks/mask_10.npy",
                full_tabular=True,  # no masking anyways so the previous path is ignored
                split="test",
                label_key=label_key,
                disease=disease,
            )
        else:
            if order == "rnd":
                tmp_name = f"/test_mask_{percent}-{{instance}}.npy"
            else:
                tmp_name = f"/test_mask_{disease}_{percent}-{{instance}}.npy"
            data = RandomMaskDatasetH5(
                h5_root="/path/to/csv_files/dir/",
                masks_path=mask_root + tmp_name,
                split="test",
                label_key=label_key,
                disease=disease,
                importance=order != "rnd",
            )
        val_loader = DataLoader(
            data, batch_size=20, shuffle=False, num_workers=8, collate_fn=collator
        )
        for data in tqdm(val_loader):
            with torch.no_grad():
                _, logits = classifier(
                    data["x"].to(device),
                    data["edge_attr"].to(device),
                    data["mask"].to(device),
                )
                preds = torch.sigmoid(logits.detach())
                label = data["label"].float()
                auc_metric.update(preds.cpu().squeeze(), label.cpu())
                f1_metric.update(preds.cpu().squeeze(), label.cpu())
        auc = auc_metric.compute()
        f1 = f1_metric.compute()
        print("Percent: ", percent, " AUC: ", auc, "F1", f1)
        # log to wandb
        wandb.log({"tabular_percent": percent, "auc_macro": auc.item(), "F1": f1})
        auc_metric.reset()


if __name__ == "__main__":
    main()
