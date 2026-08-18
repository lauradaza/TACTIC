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


from utils.data_paths import img_datas

from tactic.modeling.eva_class import EvaClass
from tactic.modeling.tabular_encoder import TabularClassifier
from tactic.modeling.baseline_models import BaselineMixer, DAFT, TIP

from tactic.dataloading.imTab_dataset import WBImgTabDataset, WBImgTabCollator

from tarte_ai import TARTE_TablePreprocessor

from trainer import BaseTrainer, init_seeds, device_config, setup, cleanup

join = os.path.join


def get_parser():
    # %% set up parser
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--project_name", type=str, default="WB_baselines")
    parser.add_argument("--task_name", type=str, default="tabular_train")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--disease", type=str, default="copd")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--merge_tokens", type=int, nargs="+", default=False)
    parser.add_argument("--no_gap", action="store_true", default=False)
    parser.add_argument("--no_mlp", action="store_true", default=False)
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="concat",
        choices=["concat", "max", "sum", "gate", "daft", "tip"],
    )

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
        full_output=args.decoder_type == "daft" or args.decoder_type == "tip",
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

    if args.decoder_type == "daft" or args.decoder_type == "tip":
        image_encoder.classifier = nn.Identity()
    else:
        image_encoder.classifier = nn.Sequential(
            nn.Linear(vis_embed_size, vis_embed_size),
            nn.GELU(),
            nn.LayerNorm(vis_embed_size),
            nn.Linear(vis_embed_size, tabular_embed_dim),
            nn.GELU(),
            nn.LayerNorm(tabular_embed_dim),
        )

    tabular_encoder = TabularClassifier(
        num_classes=args.num_classes,
        tabular_checkpoint="",
    )

    tab_ckpt = torch.load(
        args.pretrained_tab, map_location=args.device, weights_only=False
    )
    tabular_encoder.load_state_dict(tab_ckpt)
    print(f"Loaded tabular encoder from {args.pretrained_tab}")

    if args.freeze_image:
        for param in image_encoder.parameters():
            param.requires_grad = False
        for param in image_encoder.classifier.parameters():
            param.requires_grad = True

    if args.freeze_tabular:
        for param in tabular_encoder.parameters():
            param.requires_grad = False

    # finally SAM
    if args.decoder_type == "daft":
        model = DAFT(
            image_encoder=image_encoder,
            tabular_encoder=tabular_encoder,
            vis_dim=vis_embed_size,
            tab_dim=tabular_embed_dim,
            device=args.device,
            classes=args.num_classes,
        ).to(args.device)
    elif args.decoder_type == "tip":
        model = TIP(
            image_encoder=image_encoder,
            tabular_encoder=tabular_encoder,
            vis_dim=vis_embed_size,
            tab_dim=tabular_embed_dim,
            device=args.device,
            classes=args.num_classes,
        ).to(args.device)
    else:
        model = BaselineMixer(
            image_encoder=image_encoder,
            tabular_encoder=tabular_encoder,
            embed_dim=tabular_embed_dim,
            decoder=args.decoder_type,
            device=args.device,
            classes=args.num_classes,
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
        self.set_optimizer()
        self.set_lr_scheduler()

        super().__init__(model, dataloaders, args)

        self.best_auc = 0.0
        self.step_best_auc = 0.0
        self.aurocs = []
        self.f1scores = []

    def set_optimizer(self):
        model = self.model.module if self.args.multi_gpu else self.model

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, model.image_encoder.parameters()
                    )
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        model.tabular_encoder.parameters(),
                    ),
                    "lr": self.args.lr,  # * 0.1,
                },
                {
                    "params": model.tabular_decoder.parameters(),
                    "lr": self.args.lr,  # * 0.1,
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

    def init_checkpoint(self, ckp_path):
        self.start_epoch = 0
        if not ckp_path or not os.path.exists(ckp_path):
            print("Training from scratch")
            return

        if self.args.multi_gpu:
            dist.barrier()

        last_ckpt = torch.load(
            ckp_path, map_location=self.args.device, weights_only=False
        )

        strict = not self.args.allow_partial_weight
        loaded = (
            self.model.module.load_state_dict(
                last_ckpt["model_state_dict"], strict=strict
            )
            if self.args.multi_gpu
            else self.model.load_state_dict(
                last_ckpt["model_state_dict"], strict=strict
            )
        )

        self.start_epoch = last_ckpt["epoch"]
        self.latest_epoch = self.start_epoch
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
                "used_datas": img_datas,
            },
            join(self.args.save_path, f"model_{describe}.pth"),
        )

    def get_points(self, attr_dict, num_points):
        attributes, _ = attr_dict["attributes"]
        edge_attr = attr_dict["edge_attr"]
        mask = attr_dict["attn_mask"]

        if self.args.order == "rnd":
            rows, cols, _ = attributes.shape
            indices = (
                torch.multinomial(
                    torch.ones([rows, cols - 1]) / (cols - 1),
                    num_samples=num_points,
                    replacement=False,
                )
                + 1
            )
        else:
            indices = (
                torch.arange(1, num_points + 1)
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

    def interaction(self, model, image_embedding, attributes, labels):
        attr_input = self.get_points(attributes, self.args.num_attributes)
        mm_predictions, _ = model(
            image_embedding.to(self.args.device),
            attr_input,
        )
        mm_predictions = mm_predictions.squeeze(-1)
        mm_loss = self.class_loss(mm_predictions, labels.float())

        if self.args.decoder_type == "gate":
            img_predictions, _ = model(image_embedding, None)
            img_predictions = img_predictions.squeeze(-1)
            img_loss = self.class_loss(img_predictions, labels.float())
            ml_loss = torch.maximum(
                mm_loss - img_loss, torch.tensor(0.0, device=self.args.device)
            )
            mm_loss = ml_loss + mm_loss + img_loss

        return mm_predictions, mm_loss

    def train_epoch(self, epoch):
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

        self.optimizer.zero_grad()
        attr_keys = ["attributes", "edge_attr", "attn_mask"]

        loader = self.dataloaders[0]
        for step, sample in enumerate(pbar := tqdm(loader)):
            # set up data
            image = sample["data"].to(self.args.device, non_blocking=True)
            body_mask = (
                sample["body_mask"].to(self.args.device, non_blocking=True).float()
            )
            label = sample["label"].to(self.args.device, non_blocking=True).float()
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

                    prediction, loss = self.interaction(
                        model,
                        image_embedding.detach(),
                        attributes.copy(),
                        label,
                    )

                self.scaler.scale(loss / denom).backward()

            # logging and metrics
            cur_loss = loss.item()
            epoch_loss += cur_loss
            step_loss += cur_loss
            step_num += 1

            pred = torch.sigmoid(prediction.detach()).cpu()
            self.auc.update(pred, label.cpu().int())
            self.f1score.update(pred, label.cpu().int())

            pbar.set_description(f"Loss: {cur_loss:.4f}, AUC: {self.auc.compute():.4f}")

            # optimizer step
            if sync_now:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                avg_step_loss = step_loss / max(1, step_num)

                if not self.args.multi_gpu or self.args.rank == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {avg_step_loss:.4f}")
                    if avg_step_loss < self.step_best_loss:
                        self.step_best_loss = avg_step_loss

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
                self.dataloaders.sampler.set_epoch(epoch)
            epoch_loss, epoch_auc, epoch_f1 = self.train_epoch(epoch)

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
        weights = {k: [] for k in percentage}
        attr_keys = ["attributes", "edge_attr", "attn_mask"]
        labels_m = []
        for sample in tbar:
            image = sample["data"].float().to(self.args.device)
            body_mask = sample["body_mask"].float().to(self.args.device)
            labels_multi = sample["label"].type(torch.long)  # B

            labels_m.append(labels_multi)

            attributes = {k: sample[k] for k in attr_keys}
            test_attributes = attributes["attributes"][0].shape[1] - 1

            with torch.no_grad():
                image_embedding = model.image_encoder(image, body_mask)
                predictions, weight = model(
                    image_embedding,
                    None,
                )
                results[0].append(torch.sigmoid(predictions).squeeze(-1).detach().cpu())
                weights[0].append(weight)

                for perc in percentage[1:]:
                    attr_input = self.get_points(
                        attributes.copy(), int(test_attributes * perc)
                    )
                    # sparse_embeddings, _ = model.tabular_encoder(*attr_input)
                    predictions, weight = model(
                        image_embedding,
                        attr_input,
                    )
                    results[perc].append(
                        torch.sigmoid(predictions).squeeze(-1).detach().cpu()
                    )
                    weights[perc].append(weight)

        final_predictions = []
        labels_multi = torch.cat(labels_m, dim=0)
        pred_list = []
        for perc in percentage:
            preds = torch.cat(results[perc], dim=0)
            whts = sum(weights[perc]) / len(weights[perc])

            self.auc.update(preds, labels_multi)
            self.f1score.update(preds, labels_multi)

            final_predictions.append(
                {
                    "eid": "class_mode" + str(perc),
                    "AUC_multi": self.auc.compute().item(),
                    "F1_multi": self.f1score.compute().item(),
                    "Pos weight": whts,
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
    trainer.train()
    cleanup()


if __name__ == "__main__":
    args = get_parser()
    main(args)
