import os
import json
import random
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.transforms import (
    Compose,
    ClipIntensityPercentilesd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandZoomd,
    RandSpatialCropd,
    CenterSpatialCropd,
    SpatialPadd,
    EnsureTyped,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandBiasFieldd,
    RandGaussianNoised,
)

from tactic.modeling.eva_class import EvaClass
from tactic.modeling.fusion_head import FusionHead
from tactic.modeling.fusion_decoder import FusionDecoder
from tactic.modeling.tabular_encoder import TabularClassifier

from tactic.dataloading.imTab_dataset import WBImgTabDataset, WBImgTabCollator

from tarte_ai import TARTE_TablePreprocessor

from trainer import BaseTrainer, init_seeds, device_config, setup, cleanup

join = os.path.join


def get_parser():
    # %% set up parser
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--project_name", type=str, default="WB_TabularSAM")
    parser.add_argument("--task_name", type=str, default="tabular_train")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--disease", type=str, default="copd")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--merge_tokens", type=int, nargs="+", default=False)
    parser.add_argument("--num_clicks", type=int, default=10)
    parser.add_argument("--no_gap", action="store_true", default=False)
    parser.add_argument("--no_mlp", action="store_true", default=False)
    parser.add_argument("--tarte_embed", action="store_true", default=False)
    parser.add_argument("--attn_scale", type=float, default=1.0)
    # pretrained encoders
    parser.add_argument(
        "--pretrained_img",
        type=str,
        default="./work_dir/MAEm_32x32x16/model_latest.pth",
    )
    parser.add_argument(
        "--pretrained_tab",
        type=str,
        default="",
    )
    parser.add_argument("--freeze_image", action="store_true", default=False)
    parser.add_argument("--freeze_tabular", action="store_true", default=False)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_attributes", type=int, default=233)

    # training parameters
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--accumulation_steps", type=int, default=6)
    parser.add_argument("--img_size", type=int, nargs="+", default=[224, 160, 352])
    parser.add_argument("--patch_size", type=int, nargs="+", default=[32, 32, 16])

    # training device and stuff
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=12361)

    # lr scheduler and optimizer
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lr_scheduler", type=str, default="multisteplr")
    parser.add_argument("--step_size", type=list, default=[60, 90])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # test
    parser.add_argument("--order", type=str, default="rnd", choices=["rnd", "hi", "li"])

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
    args.log_out_dir = join(args.work_dir, args.task_name)
    args.save_path = join(args.work_dir, args.task_name)
    os.makedirs(args.save_path, exist_ok=True)
    return args


class TabularFeatures(TabularClassifier):
    def __init__(self, num_classes, just_embed, embed_dim, pretrained, device):
        super().__init__(num_classes, "", just_embed=just_embed)
        # TODO multiprocess
        if pretrained:
            state_dict = torch.load(pretrained, weights_only=True)
            model_dict = self.state_dict()
            filtered_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            self.load_state_dict(filtered_dict, strict=False)
        self.embed_dim = embed_dim
        self.tabular_head = nn.Identity()  # classifier
        self.module_device = device

    def forward(self, x, edge_attr=None, mask=None):
        if x is None:
            return torch.empty((1, 0, self.embed_dim), device=self.module_device)
        tabular_features = self.tabular_encoder(x, edge_attr, mask)
        return tabular_features


def build_model(args):
    vis_embed_size = 864
    tabular_embed_dim = 768

    image_encoder = EvaClass(
        input_channels=2,
        embed_dim=vis_embed_size,
        patch_embed_size=args.patch_size,
        output_channels=args.num_classes,
        num_register_tokens=1,  # class token
        eva_depth=16,
        eva_numheads=12,
        input_shape=args.img_size,
        drop_path_rate=0.2,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=0.4,
        merge_tokens=args.merge_tokens,
        full_output=True,
    ).to(args.device)

    img_ckpt = torch.load(
        args.pretrained_img, map_location=args.device, weights_only=False
    )

    model_dict = image_encoder.state_dict()
    filtered_dict = {
        k: v
        for k, v in img_ckpt["model_state_dict"].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    image_encoder.load_state_dict(filtered_dict, strict=False)

    # Project to tabular embeding dimension
    image_encoder.classifier = nn.Sequential(
        nn.Linear(vis_embed_size, vis_embed_size),
        nn.GELU(),
        nn.LayerNorm(vis_embed_size),
        nn.Linear(vis_embed_size, tabular_embed_dim),
        nn.GELU(),
        nn.LayerNorm(tabular_embed_dim),
    )

    tabular_encoder = TabularFeatures(
        num_classes=args.num_classes,
        just_embed=args.tarte_embed,
        embed_dim=tabular_embed_dim,
        pretrained=args.pretrained_tab,
        device=args.device,
    )

    if args.freeze_image:
        for param in image_encoder.parameters():
            param.requires_grad = False
        for param in image_encoder.classifier.parameters():
            param.requires_grad = True

    if args.freeze_tabular:
        for param in tabular_encoder.parameters():
            param.requires_grad = False

    # Use the name transformation and initial mapping from TARTE
    processor = TARTE_TablePreprocessor()
    processor._load_lm_model()

    out_disease = (
        args.disease
        if args.disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )
    out_disease = out_disease.replace("_", " ")
    class_names = [
        "No presence of " + out_disease + " detected",
        out_disease + " diagnosed",
    ]

    print("Output tokens:", class_names)

    embb_names = torch.tensor(processor._transform_names(class_names))
    embb = tabular_encoder.tabular_encoder.initial_e(embb_names)

    # finally SAM
    model = FusionHead(
        image_encoder=image_encoder,
        tabular_encoder=tabular_encoder,
        tabular_decoder=FusionDecoder(
            transformer_dim=tabular_embed_dim,
            cad_head_depth=3,
            cad_head_hidden_dim=256,
            init_embeddings=embb,
            attn_scale=args.attn_scale,
        ),
    ).to(args.device)
    if args.multi_gpu:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    return model


def get_dataloaders(args):
    if args.freeze_image:
        train_transform = Compose(
            [
                EnsureChannelFirstd(
                    keys=["img", "mask"], channel_dim=0
                ),  # [1, D, H, W]
                ClipIntensityPercentilesd(keys=["img"], lower=1, upper=99),
                NormalizeIntensityd(
                    keys=["img"],
                    channel_wise=True,  # per patient, per channel
                ),
                RandZoomd(
                    keys=["img", "mask"],
                    prob=1.0,
                    min_zoom=0.8,  # original is 0.2
                    max_zoom=1.2,  # original is 1.0
                    mode=["trilinear", "nearest"],
                    align_corners=[False, None],
                    padding_mode="constant",
                ),
                RandSpatialCropd(
                    keys=["img", "mask"],
                    roi_size=args.img_size,
                    random_size=False,
                ),
                SpatialPadd(
                    keys=["img", "mask"],
                    spatial_size=args.img_size,
                    mode="constant",
                    constant_values=0,
                ),
                EnsureTyped(keys=["img", "mask"], track_meta=False),
            ]
        )
    else:
        train_transform = Compose(
            [
                EnsureChannelFirstd(keys=["img", "mask"], channel_dim=0),
                ClipIntensityPercentilesd(keys=["img"], lower=1, upper=99),
                NormalizeIntensityd(
                    keys=["img"],
                    channel_wise=True,  # per patient, per channel
                ),
                RandSpatialCropd(
                    keys=["img", "mask"],
                    roi_size=args.img_size,
                    random_center=True,
                    random_size=False,
                ),
                RandAffined(
                    keys=["img", "mask"],
                    prob=0.3,
                    rotate_range=(0.05, 0.05, 0.05),
                    scale_range=(0.05, 0.05, 0.05),
                    mode=["bilinear", "nearest"],
                    padding_mode="border",
                ),
                SpatialPadd(
                    keys=["img", "mask"],
                    spatial_size=args.img_size,
                    mode="constant",
                    constant_values=0,
                ),
                RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.8),
                RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.8),
                RandBiasFieldd(keys=["img"], prob=0.5, coeff_range=(0.0, 0.3)),
                RandGaussianNoised(keys=["img"], prob=0.3, std=0.01),
                EnsureTyped(keys=["img", "mask"], track_meta=False),
            ]
        )

    label_key = (
        args.disease.capitalize().replace("_", " ")
        if args.disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )
    train_dataset = WBImgTabDataset(
        root_dir="/path/to/img/data",
        csv_path="/path/to/csv_files/dir/",
        transform=train_transform,
        split="balanced train",
        label_key=label_key,
        disease=args.disease,
    )

    tarte_tab_prepper = TARTE_TablePreprocessor()
    args.num_attributes = train_dataset.tabular_data.shape[1]
    tarte_tab_prepper.fit(train_dataset.tabular_data)
    collator = WBImgTabCollator(tarte_tab_prepper)

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if args.multi_gpu else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collator,
        # prefetch_factor=2,
    )

    test_transform = Compose(
        [
            EnsureChannelFirstd(keys=["img", "mask"], channel_dim=0),  # [1, D, H, W]
            ClipIntensityPercentilesd(keys=["img"], lower=1, upper=99),
            NormalizeIntensityd(
                keys=["img"],
                channel_wise=True,  # per patient, per channel
            ),
            CenterSpatialCropd(
                keys=["img", "mask"],
                roi_size=args.img_size,
            ),
            SpatialPadd(
                keys=["img", "mask"],
                spatial_size=args.img_size,
                mode="constant",
                constant_values=0,
            ),
            EnsureTyped(keys=["img", "mask"], track_meta=False),
        ]
    )

    test_dataset = WBImgTabDataset(
        root_dir="/path/to/img/data",
        csv_path="/path/to/csv_files/dir/",
        transform=test_transform,
        split="test",
        label_key=label_key,
        disease=args.disease,
    )

    if args.order == "rnd":
        rnd_file = "/path/to/csv_files/dir/rnd_masks/mask_100.npy"
        eval_order = np.load(rnd_file, allow_pickle=True).item()
    else:
        imp_file = (
            f"/home/iml/laura.daza/MedSAM/results_tab/{args.disease}_importance.json"
        )
        with open(imp_file, "r") as f:
            importance = json.load(f)
        if args.order == "hi":
            eval_order = importance["attributes"]
        else:  # low importance
            eval_order = importance["attributes"][::-1]

    test_collator = WBImgTabCollator(tarte_tab_prepper, order=eval_order)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator,
    )
    return train_loader, test_loader


class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, args):
        self.args = args
        self.model = model
        self.dataloaders = dataloaders
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        super().__init__(model, dataloaders, args)

        self.best_auc = 0.0
        self.aurocs = []
        self.f1scores = []
        self.class_loss = nn.CrossEntropyLoss()

    def set_optimizer(self):
        model = self.model.module if self.args.multi_gpu else self.model

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.image_encoder.parameters(),
                },
                {
                    "params": model.tabular_encoder.parameters(),
                    "lr": self.args.lr,  # * 0.1,
                },
                {
                    "params": model.tabular_decoder.parameters(),
                    "lr": self.args.lr,  # * 0.1,
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )

    def init_checkpoint(self, ckp_path):
        self.start_epoch = 0
        self.latest_epoch = 0
        if not ckp_path or not os.path.exists(ckp_path):
            print("Training from scratch")
            return

        if self.args.multi_gpu:
            dist.barrier()

        last_ckpt = torch.load(
            ckp_path, map_location=self.args.device, weights_only=False
        )

        strict = not self.args.allow_partial_weight
        if self.args.multi_gpu:
            loaded = self.model.module.load_state_dict(
                last_ckpt["model_state_dict"], strict=strict
            )
        else:
            loaded = self.model.load_state_dict(
                last_ckpt["model_state_dict"], strict=strict
            )

        self.start_epoch = last_ckpt["epoch"]
        self.latest_epoch = last_ckpt["epoch"]
        if self.args.resume and not self.args.eval:
            self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(last_ckpt["lr_scheduler_state_dict"])
            self.losses = last_ckpt.get("losses", [])
            self.aurocs = last_ckpt.get("AUCs", [])
            self.f1scores = last_ckpt.get("f1scores", [])
            self.best_loss = last_ckpt.get("best_loss", np.inf)
            self.best_auc = last_ckpt.get("best_auc", np.inf)

        print(
            f"Loaded weights from {ckp_path} (epoch {self.start_epoch}) with strict={strict}"
        )
        print("load result:", loaded)

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "AUCs": self.aurocs,
                "f1scores": self.f1scores,
                "best_loss": self.best_loss,
                "best_auc": self.best_auc,
                "args": self.args,
            },
            join(self.args.save_path, f"model_{describe}.pth"),
        )

    def batch_forward(self, model, image_embedding, attributes):
        sparse_embeddings = model.tabular_encoder(*attributes)
        predictions = model.tabular_decoder(
            image_embeddings=image_embedding.to(self.args.device),
            sparse_prompt_embeddings=sparse_embeddings,
        )
        return predictions

    def get_points(self, attr_dict, num_points):
        attributes, _ = attr_dict["attributes"]
        edge_attr = attr_dict["edge_attr"]
        mask = attr_dict["attn_mask"]

        # TARTE's [CLS] token is in position 0 and we always include it
        if self.args.order == "rnd":
            # cols includes [CLS] token
            rows, cols, _ = attributes.shape
            indices = (
                torch.multinomial(
                    torch.ones([rows, cols - 1]) / (cols - 1),
                    num_samples=num_points,
                    replacement=False,
                )
                + 1
            )  # +1 to avoid [CLS] token
            # add the [CLS] token now
            indices = torch.cat([torch.ones(rows, 1).long(), indices], dim=1)
        else:
            # num_points + 1 to include the [CLS] token
            indices = (
                torch.arange(0, num_points + 1)
                .long()
                .unsqueeze(0)
                .repeat(mask.shape[0], 1)
            )

        row_idx = torch.arange(attributes.size(0)).unsqueeze(1)
        values = attributes[row_idx, indices]
        edge_attr = edge_attr[row_idx, indices]
        mask = mask[row_idx, indices]

        attr_input = values.to(self.args.device)
        edge_attr = edge_attr.to(self.args.device)
        mask = mask.to(self.args.device)

        return attr_input, edge_attr, mask

    def interaction(self, model, image_embedding, attributes, labels, num_clicks):
        return_loss = 0
        num_points = np.random.randint(1, self.args.num_attributes)  # random in2
        for _ in range(num_clicks):
            # random in
            attr_input = self.get_points(attributes, num_points)
            mm_predictions = self.batch_forward(
                model,
                image_embedding,
                attributes=attr_input,
            )
            mm_predictions = mm_predictions.squeeze(-1)
            mm_loss = self.class_loss(mm_predictions, labels.float())
            return_loss += mm_loss / num_clicks

        img_predictions = self.batch_forward(
            model, image_embedding, attributes=[None]
        ).squeeze(-1)
        img_loss = self.class_loss(img_predictions, labels.float())
        return mm_predictions, return_loss, img_loss

    def train_epoch(self, epoch, num_clicks):
        device = self.args.device
        self.model.train()
        self.auc.reset()
        self.f1score.reset()

        epoch_loss = 0
        step_loss = 0
        step_num = 0

        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model

        self.optimizer.zero_grad(set_to_none=True)
        attr_keys = ["attributes", "edge_attr", "attn_mask"]

        loader = self.dataloaders[0]
        for step, sample in enumerate(pbar := tqdm(loader)):
            # set up data
            image = sample["data"].to(device)
            body_mask = sample["body_mask"].to(device).float()
            label = sample["label"].long()
            label = F.one_hot(label, num_classes=2).float().to(device)
            attributes = {k: sample[k] for k in attr_keys}  # B, 1, 234

            # determine if we should sync gradients now
            is_update_step = (step + 1) % self.args.accumulation_steps == 0
            is_last_step = step + 1 == len(loader)
            sync_now = is_update_step or is_last_step

            remaining = len(loader) - step
            denom = min(self.args.accumulation_steps, remaining)

            context = (
                nullcontext
                if (not self.args.multi_gpu or sync_now)
                else self.model.no_sync
            )

            # forward pass
            with context():
                with torch.amp.autocast("cuda"):
                    image_embedding = model.image_encoder(image, body_mask)

                    prediction, mm_loss, mmi_loss = self.interaction(
                        model,
                        image_embedding.detach(),
                        attributes.copy(),
                        label,
                        num_clicks=num_clicks,
                    )
                    ml_loss = torch.clamp(mm_loss - mmi_loss, min=0.0)
                    loss = ml_loss + mm_loss + mmi_loss

                self.scaler.scale(loss / denom).backward()

            # logging and metrics
            cur_loss = loss.item()
            epoch_loss += cur_loss
            step_loss += cur_loss
            step_num += 1

            pred = torch.softmax(prediction.detach(), dim=1).cpu()[:, 1]
            label = label.cpu().int()[:, 1]
            self.auc.update(pred, label)
            self.f1score.update(pred, label)

            pbar.set_description(f"Loss: {cur_loss:.4f}, AUC: {self.auc.compute():.4f}")

            # optimizer step
            if sync_now:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                print_loss = step_loss / max(1, step_num)
                print_auc = self.auc.compute().item()

                if not self.args.multi_gpu or self.args.rank == 0:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Loss: {print_loss:.4f}, AUC: {print_auc}"
                    )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

                step_loss = 0.0
                step_num = 0

        epoch_loss /= max(1, len(loader))
        epoch_auc = self.auc.compute().item()
        epoch_f1 = self.f1score.compute().item()
        self.wandb.log(
            {"epoch": epoch, "train_loss": epoch_loss, "train_AUC": epoch_auc}
        )
        return epoch_loss, epoch_auc, epoch_f1

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        # training loop
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders[0].sampler.set_epoch(epoch)
            num_clicks = self.args.num_clicks
            epoch_loss, epoch_auc, epoch_f1 = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            # logging and checkpointing
            if not self.args.multi_gpu or self.args.rank == 0:
                self.losses.append(epoch_loss)
                self.aurocs.append(epoch_auc)
                self.f1scores.append(epoch_f1)
                print(f"EPOCH: {epoch}, Loss: {epoch_loss}")
                print(f"EPOCH: {epoch}, AUC: {epoch_auc}, F1: {epoch_f1}")

                state_dict = (
                    self.model.module.state_dict()
                    if self.args.multi_gpu
                    else self.model.state_dict()
                )

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe="latest")
                self.latest_epoch = epoch + 1

                self.plot_result(self.losses, "Cross Entropy Loss", "Loss")
                self.plot_result(self.aurocs, "AUC", "AUC")
                self.plot_result(self.f1scores, "F1 score", "F1 score")
        print("Training done!")

    def eval(self):
        model = self.model
        model.eval()
        self.auc.reset()
        self.f1score.reset()

        tbar = tqdm(self.dataloaders[1])  # test

        percentage = np.linspace(0, 1, 11)
        results = {k: [] for k in percentage}
        attr_keys = ["attributes", "edge_attr", "attn_mask"]
        labels_m = []
        for sample in tbar:
            image = sample["data"].float().to(self.args.device)
            body_mask = sample["body_mask"].float().to(self.args.device)
            labels_multi = sample["label"].type(torch.long)  # B
            labels_multi = F.one_hot(labels_multi, num_classes=2).float()

            labels_m.append(labels_multi[:, 1])

            attributes = {k: sample[k] for k in attr_keys}
            test_attributes = attributes["attributes"][0].shape[1] - 1

            with torch.no_grad():
                image_embedding = model.image_encoder(image, body_mask)
                sparse_embeddings = model.tabular_encoder(None)
                predictions = model.tabular_decoder(
                    image_embeddings=image_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                )
                results[0].append(
                    torch.softmax(predictions, dim=1).detach().cpu()[:, 1]
                )

                for perc in percentage[1:]:
                    attr_input = self.get_points(
                        attributes.copy(), int(test_attributes * perc)
                    )
                    sparse_embeddings = model.tabular_encoder(*attr_input)
                    predictions = model.tabular_decoder(
                        image_embeddings=image_embedding,
                        sparse_prompt_embeddings=sparse_embeddings,
                    )
                    results[perc].append(
                        torch.softmax(predictions, dim=1).detach().cpu()[:, 1]
                    )

        final_predictions = []
        labels_multi = torch.cat(labels_m, dim=0)
        if len(labels_multi.shape) == 1:
            labels_multi = labels_multi.unsqueeze(1)
        pred_list = []
        for perc in percentage:
            preds = torch.cat(results[perc], dim=0)

            self.auc.update(preds, labels_multi)
            self.f1score.update(preds, labels_multi)

            final_predictions.append(
                {
                    "eid": "class_mode" + str(perc),
                    "AUC_multi": self.auc.compute().item(),
                    "F1_multi": self.f1score.compute().item(),
                }
            )

            self.auc.reset()
            self.f1score.reset()
            pred_list.append(preds.numpy())

        print(final_predictions)
        df = pd.DataFrame(final_predictions)
        df.to_csv(
            join(
                self.args.log_out_dir,
                f"results_{self.latest_epoch}_{self.args.order}.csv",
            ),
            index=False,
        )

        labels_multi = labels_multi.numpy()
        df = pd.DataFrame(np.concatenate([labels_multi, *pred_list], axis=1))
        df.to_csv(
            join(
                self.args.log_out_dir,
                f"preds_{self.latest_epoch}_{self.args.order}.csv",
            ),
            index=False,
        )

        print("Evaluation done!")


def main(args):
    mp.set_sharing_strategy("file_system")
    device_config(args)

    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = Trainer(model, dataloaders, args)
        # Train
        if not args.eval:
            trainer.train()
        # Eval
        trainer.eval()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = max(1, int(args.num_workers / args.ngpus_per_node))
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode="w",
        filename=os.path.join(args.log_out_dir, f"output_{cur_time}.log"),
    )

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = Trainer(model, dataloaders, args)

    if not args.eval:
        trainer.train()
    trainer.eval()

    cleanup()


if __name__ == "__main__":
    args = get_parser()
    main(args)
