import os
import random
import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# import monai
from monai.transforms import (
    Compose,
    ClipIntensityPercentilesd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandZoomd,
    RandSpatialCropd,
    SpatialPadd,
    EnsureTyped,
)

from tactic.modeling.evaMAE import EvaMAE
from tactic.nnunet.primus import LayerNormNd
from tactic.dataloading.pretrain_dataset import WBMRI_Dataset
from tactic.modeling.contrastive_loss import PretrainLossMultiSequence

from trainer import BaseTrainer, init_seeds, device_config, setup, cleanup

join = os.path.join


def get_parser():
    # %% set up parser
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--project_name", type=str, default="WBMRI")
    parser.add_argument("--task_name", type=str, default="union_train")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)
    parser.add_argument("--merge_tokens", type=int, nargs="+", default=False)
    parser.add_argument("--masking", type=float, default=0.75)
    parser.add_argument("--decoder_depth", type=int, default=2)

    # training parameters
    parser.add_argument("--batch_size", type=int, default=12)  # original MAE: 4096
    parser.add_argument("--accumulation_steps", type=int, default=20)
    parser.add_argument("--img_size", type=int, nargs="+", default=[224, 160, 352])
    parser.add_argument("--patch_size", type=int, nargs="+", default=[32, 32, 16])

    # training device and stuff
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=12361)

    # lr scheduler and optimizer
    parser.add_argument(
        "--lr", type=float, default=8e-4
    )  # original MAE: 0.0024 (2.4e-3)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)  # 5% of total epochs

    args = parser.parse_args()

    args.eval = False
    args.log_out_dir = join(args.work_dir, args.task_name)
    args.save_path = join(args.work_dir, args.task_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
    os.makedirs(args.save_path, exist_ok=True)
    return args


def build_model(args):
    model = EvaMAE(
        input_channels=1,
        embed_dim=864,
        patch_embed_size=args.patch_size,
        output_channels=1,
        encoder_eva_depth=16,
        encoder_eva_numheads=12,
        decoder_eva_depth=args.decoder_depth,  # has always been 2!
        decoder_eva_numheads=12,
        input_shape=args.img_size,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        drop_path_rate=0.2,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=args.masking,
        merge_tokens=args.merge_tokens,
    ).to(args.device)

    if args.multi_gpu:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    return model


def get_dataloaders(args):
    train_transform = Compose(
        [
            EnsureChannelFirstd(keys=["img", "mask"], channel_dim=0),  # [1, D, H, W]
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

    dataset = WBMRI_Dataset(
        root_dir="/path/to/img/data",
        transform=train_transform,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if args.multi_gpu else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        prefetch_factor=2,
    )
    return [loader]


class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, args):
        super().__init__(model, dataloaders, args)

    def set_loss_fn(self):
        self.pretrain_loss = PretrainLossMultiSequence()

    def train_epoch(self, epoch):
        # set up model and metrics
        self.model.train()
        epoch_loss = 0
        step_loss = 0
        step_num = 0

        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model

        self.optimizer.zero_grad()

        # manual LR adjustment per step
        self.adjust_learning_rate(self.optimizer, epoch, self.args)

        # training loop
        loader = self.dataloaders[0]  # usually 0: train, 1: eval
        for step, sample in enumerate(pbar := tqdm(loader)):
            # set up data
            image = sample["data"].to(self.args.device, non_blocking=True)
            body_mask = sample["body_mask"].to(self.args.device, non_blocking=True)

            B, C, D, H, W = image.shape
            image = image.reshape(B * C, 1, D, H, W)

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

            with context():
                with torch.amp.autocast("cuda"):
                    output = model(image, mask=body_mask)

                    loss = self.pretrain_loss(
                        output["pooled"],
                        output["rec"],
                        output["body_mask"],
                        output["mae_mask"],
                    )

                self.scaler.scale(loss["loss"] / denom).backward()

            cur_loss = loss["loss"].item()
            epoch_loss += cur_loss
            step_loss += cur_loss
            step_num += 1

            if sync_now:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                new_lr = self.adjust_learning_rate(
                    self.optimizer,
                    step / max(1, len(loader)) + epoch,
                    self.args,
                )

                avg_step_loss = step_loss / max(1, step_num)
                self.wandb.log(
                    {"epoch": epoch, "train_loss": avg_step_loss, "lr": new_lr}
                )
                step_loss = 0

                if not self.args.multi_gpu or self.args.rank == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {avg_step_loss:.4f}")
                    if avg_step_loss < self.step_best_loss:
                        self.step_best_loss = avg_step_loss

                step_loss = 0.0
                step_num = 0

        epoch_loss /= max(1, len(loader))
        return epoch_loss


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
        trainer.train()


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
        filename=join(args.log_out_dir, f"output_{cur_time}.log"),
    )

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = Trainer(model, dataloaders, args)
    trainer.train()
    cleanup()


if __name__ == "__main__":
    args = get_parser()
    main(args)
