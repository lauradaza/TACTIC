import os
import random
import logging
import argparse
import datetime
import numpy as np

import torch

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
)

from tactic.nnunet.eva_class import EvaClass
from tactic.dataloading.img_dataset import WBMRI_train_Dataset

from trainer import BaseTrainer, init_seeds, device_config, setup, cleanup

join = os.path.join


def get_parser():
    # %% set up parser
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--project_name", type=str, default="WBMRI_class")
    parser.add_argument("--task_name", type=str, default="union_train")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--disease", type=str, default="copd")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow_partial_weight", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--merge_tokens", type=int, nargs="+", default=False)
    parser.add_argument("--num_classes", type=int, default=1)

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
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=2)  # 5% of total epochs

    args = parser.parse_args()

    args.log_out_dir = join(args.work_dir, args.task_name)
    args.save_path = join(args.work_dir, args.task_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])
    os.makedirs(args.save_path, exist_ok=True)
    return args


def build_model(args):
    model = EvaClass(
        input_channels=2,
        embed_dim=864,
        patch_embed_size=args.patch_size,
        output_channels=args.num_classes,
        num_register_tokens=1,  # class token
        eva_depth=16,
        eva_numheads=12,
        input_shape=args.img_size,
        drop_path_rate=0.2,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=0.4,  # roughly 40% of patches are bg
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

    label_key = (
        args.disease.capitalize().replace("_", " ")
        if args.disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )

    train_dataset = WBMRI_train_Dataset(
        root_dir="/path/to/img/data",
        transform=train_transform,
        split="balanced train",
        label_key=label_key,
        disease=args.disease,
    )

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
        prefetch_factor=2,
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

    test_dataset = WBMRI_train_Dataset(
        root_dir="/path/to/img/data",
        transform=test_transform,
        split="test",
        label_key=label_key,
        disease=args.disease,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, test_loader


class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, args):
        super().__init__(model, dataloaders, args)
        self.set_lr_scheduler()

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

        if "citations" in last_ckpt:
            print("This model was trained with nnUNet *sigh*")
            self.args.allow_partial_weight = True
            new_dict = {}
            for k, v in last_ckpt["network_weights"].items():
                if k.startswith("up_projection"):
                    continue
                new_dict["image_encoder." + k] = v
            last_ckpt["model_state_dict"] = new_dict
            del last_ckpt["network_weights"]

        resume = self.args.resume or self.args.eval
        model_dict = (
            self.model.state_dict()
            if not self.args.multi_gpu
            else self.model.module.state_dict()
        )
        if self.args.checkpoint and not resume:
            checkpoint_dict = last_ckpt["model_state_dict"]
            filtered_dict = {
                k: v
                for k, v in checkpoint_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            mods = model_dict["down_projection.merge.weight"].shape[0]
            if mods != 1:
                filtered_dict["down_projection.merge.weight"] = checkpoint_dict[
                    "down_projection.merge.weight"
                ].repeat(mods, mods, 1, 1, 1)
                filtered_dict["down_projection.merge.bias"] = checkpoint_dict[
                    "down_projection.merge.bias"
                ].repeat(mods)
                filtered_dict["down_projection.proj.weight"] = checkpoint_dict[
                    "down_projection.proj.weight"
                ].repeat(1, mods, 1, 1, 1)
            strict = False
            self.start_epoch = 0
        else:
            filtered_dict = last_ckpt["model_state_dict"]
            strict = not self.args.allow_partial_weight
            self.start_epoch = last_ckpt["epoch"]

        loaded = (
            self.model.module.load_state_dict(filtered_dict, strict=strict)
            if self.args.multi_gpu
            else self.model.load_state_dict(filtered_dict, strict=strict)
        )

        if self.args.resume and not self.args.eval:
            self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            self.losses = last_ckpt.get("losses", [])
            self.best_loss = last_ckpt.get("best_loss", np.inf)

        self.latest_epoch = self.start_epoch

        print(
            f"Loaded weights from {ckp_path} (epoch {self.start_epoch}) with strict={strict}"
        )
        print("load result:", loaded)

    def train_epoch(self, epoch):
        # set up model and metrics
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

        # training loop
        loader = self.dataloaders[0]  # usually 0: train, 1: eval
        for step, sample in enumerate(pbar := tqdm(loader)):
            # set up data
            image = sample["data"].to(self.args.device, non_blocking=True)
            body_mask = (
                sample["body_mask"].to(self.args.device, non_blocking=True).float()
            )
            label = sample["label"].to(self.args.device, non_blocking=True).float()

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
                    prediction = model(image, body_mask)
                    loss = self.class_loss(prediction, label)

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
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                avg_step_loss = step_loss / max(1, step_num)

                if not self.args.multi_gpu or self.args.rank == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {avg_step_loss:.4f}")
                    if avg_step_loss < self.step_best_loss:
                        self.step_best_loss = avg_step_loss

                step_loss = 0.0
                step_num = 0

        epoch_loss /= max(1, len(loader))
        self.wandb.log({"epoch": epoch, "Loss": epoch_loss, "AUC": self.auc.compute()})
        return epoch_loss

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        # training loop
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")

            if self.args.multi_gpu:
                dist.barrier()
                if hasattr(self.dataloaders[0], "sampler") and hasattr(
                    self.dataloaders[0].sampler, "set_epoch"
                ):
                    self.dataloaders[0].sampler.set_epoch(epoch)

            epoch_loss = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            # logging and checkpointing
            if not self.args.multi_gpu or self.args.rank == 0:
                self.losses.append(epoch_loss)
                print(f"EPOCH: {epoch}, Loss: {epoch_loss}")

                state_dict = (
                    self.model.module.state_dict()
                    if self.args.multi_gpu
                    else self.model.state_dict()
                )

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe="latest")
                self.latest_epoch = epoch + 1

                self.plot_result(self.losses, "MSE", "Loss")
        self.wandb.finish()

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": self.losses,
                "best_loss": self.best_loss,
                "args": self.args,
            },
            join(self.args.save_path, f"model_{describe}.pth"),
        )


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
        filename=join(args.log_out_dir, f"output_{cur_time}.log"),
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
