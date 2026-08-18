import os
import click

import wandb
import random
import numpy as np

# import pandas as pd
from tqdm import tqdm

from tarte_ai import TARTE_TablePreprocessor

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, F1Score

from tactic.dataloading.tab_dataset import (
    MRClassificationDatasetH5,
    ClassificationCollator,
)

from tactic.modeling.tabular_encoder import TabularClassifier

ROOT_PATH = "./results_tab"


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
@click.option("--batch_size", "-b", help="Traning batch size.", default=128, type=int)
@click.option("--epochs", "-e", help="Number of epochs.", default=100, type=int)
@click.option(
    "--store",
    "-s",
    help="Where you want to store the models and results.",
    required=True,
    type=str,
)
@click.option(
    "--frozen",
    "-f",
    help="Whether the backbone should be frozen.",
    required=False,
    default=True,
    type=bool,
)
@click.option(
    "--model",
    "-m",
    required=False,
    # default="./results_tab/fixed_attr/multilabel_cad/last.pth",
    default="",
)
@click.option(
    "--disease",
    "-d",
    required=False,
    default="cancer",
)
@click.option("--num_workers", "-w", required=False, default=16)
@click.option("--loss_weight", "-l", required=False, default=1, type=float)
def main(
    batch_size,
    epochs,
    store,
    frozen,
    model,
    num_workers,
    loss_weight,
    disease,
):
    init_seeds()

    num_classes = 1

    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "multilabel_cad")
    os.makedirs(classifier_folder, exist_ok=True)

    wandb.init(project="CVD_RISK", dir=store, name=store.split("/")[-1])

    label_key = (
        disease.capitalize().replace("_", " ")
        if disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )

    train_data = MRClassificationDatasetH5(
        h5_root="/path/to/csv_files/dir/",
        instance="2",
        split="balanced train",
        simulate_missing=True,
        label_key=label_key,
        disease=disease,
    )

    val_data = MRClassificationDatasetH5(
        h5_root="/path/to/csv_files/dir/",
        instance="2",
        split="test",
        label_key=label_key,
        disease=disease,
    )

    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_prepper.fit(train_data.tabular_data)
    collator = ClassificationCollator(tarte_tab_prepper)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collator,
    )

    classifier = TabularClassifier(num_classes=num_classes, tabular_checkpoint=model)

    classifier.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("MULTIPLE CUDA DEVICES")
            classifier = torch.nn.DataParallel(classifier)
            torch.backends.cudnn.benchmark = True
        classifier.to(device)
    else:
        device = torch.device("cpu")

    lr = 1e-5

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    auc_train = AUROC(task="binary")
    auc_val_tab = AUROC(task="binary")

    for epoch in range(epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        classifier.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            b = data["x"].shape[0] // 2
            _, logits = classifier(
                data["x"].to(device),
                data["edge_attr"].to(device),
                data["mask"].to(device),
            )
            preds = torch.sigmoid(logits.detach())
            auc_train.update(preds.cpu(), data["label"].cpu().long())
            loss_more = criterion(
                logits[:b].squeeze(), data["label"][:b].to(device).squeeze().float()
            )
            loss_less = criterion(
                logits[b:].squeeze(), data["label"][b:].to(device).squeeze().float()
            )
            ml_loss = torch.maximum(
                loss_more - loss_less, torch.tensor(0.0, device=device)
            )
            loss = loss_more + loss_less + ml_loss
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        auc_train_metric = auc_train.compute()
        auc_train.reset()
        wandb.log(
            {
                "epoch_train_loss: ": np.mean(np.array(epoch_loss)),
                "train_auc": auc_train_metric,
            }
        )

        classifier.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                b = data["x"].shape[0] // 2
                _, logits = classifier(
                    data["x"][:b].to(device),
                    data["edge_attr"][:b].to(device),
                    data["mask"][:b].to(device),
                )
                val_loss = criterion(
                    logits[:b].squeeze(), data["label"][:b].to(device).squeeze().long()
                )
                preds = torch.sigmoid(logits.detach())
                auc_val_tab.update(
                    preds[:b].cpu().squeeze(), data["label"][:b].cpu().long().squeeze()
                )
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})

        torch.save(classifier.state_dict(), os.path.join(classifier_folder, "last.pth"))

        auc_tab_metric = auc_val_tab.compute()
        wandb.log(
            {
                "epoch_val_loss": np.mean(epoch_val_loss),
                "auc_tab": auc_tab_metric.item(),
            }
        )

        torch.save(
            classifier.state_dict(),
            os.path.join(classifier_folder, "epoch" + str(epoch) + ".pth"),
        )

        auc_val_tab.reset()

    print("SUCCESS")


if __name__ == "__main__":
    main()
