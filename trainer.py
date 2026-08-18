import os
import json
import math
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from torchmetrics import AUROC, F1Score

import torch
import torch.nn as nn

from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_GIT"] = "true"

import wandb

join = os.path.join


def get_parser():
    return None


def build_model(args):
    print("Create your model builder")


def get_dataloaders(args):
    print("Create your dataloaders")


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.args = args
        self.model = model
        self.dataloaders = dataloaders
        self.best_loss = np.inf
        self.step_best_loss = np.inf
        self.losses = []
        self.set_loss_fn()
        self.set_optimizer()
        if args.resume or args.eval:
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, "model_latest.pth")
            )
        else:
            self.init_checkpoint(self.args.checkpoint)

        if (
            not args.eval
            and self.args.task_name != "debug"
            and (not args.multi_gpu or args.rank == 0)
        ):
            run_id_file = join(args.log_out_dir, "wandb_resume.json")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r") as f:
                    data = json.load(f)
                run_id = data["run_id"]
            else:
                run_id = wandb.util.generate_id()
                with open(run_id_file, "w") as f:
                    json.dump({"run_id": run_id}, f)

            self.wandb = wandb.init(
                project=args.project_name,
                name=args.task_name,
                id=run_id,
                dir=args.log_out_dir,
                resume="allow",
                config={
                    "learning_rate": args.lr,
                    "epochs": args.num_epochs,
                    "batch_size": args.batch_size * args.accumulation_steps,
                },
            )

    def set_loss_fn(self):
        self.class_loss = nn.BCEWithLogitsLoss()
        self.auc = AUROC(task="binary")
        self.f1score = F1Score(task="binary")

    def set_optimizer(self):
        model = self.model.module if self.args.multi_gpu else self.model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

    def adjust_learning_rate(self, optimizer, epoch, args):
        """LR adjustment per step with linear warmup and cosine decay"""
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / max(1, args.warmup_epochs)
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / max(1, (args.num_epochs - args.warmup_epochs))
                )
            )
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

    def set_lr_scheduler(self):
        """LR scheduler (per epoch)"""
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.args.step_size, self.args.gamma
            )
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.args.step_size[0], self.args.gamma
            )
        elif self.args.lr_scheduler == "coswarm":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

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

        if self.args.resume and not self.args.eval:
            self.start_epoch = last_ckpt["epoch"]
            self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            self.losses = last_ckpt.get("losses", [])
            self.best_loss = last_ckpt.get("best_loss", np.inf)

        self.latest_epoch = self.start_epoch

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
                "losses": self.losses,
                "best_loss": self.best_loss,
                "args": self.args,
            },
            join(self.args.save_path, f"model_{describe}.pth"),
        )

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

        # manual LR adjustment per step
        self.adjust_learning_rate(self.optimizer, epoch, self.args)

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

                new_lr = self.adjust_learning_rate(
                    self.optimizer,
                    step / max(1, len(loader)) + epoch,
                    self.args,
                )

                avg_step_loss = step_loss / max(1, step_num)
                self.wandb.log(
                    {"epoch": epoch, "train_loss": avg_step_loss, "lr": new_lr}
                )

                if not self.args.multi_gpu or self.args.rank == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {avg_step_loss:.4f}")
                    if avg_step_loss < self.step_best_loss:
                        self.step_best_loss = avg_step_loss

                step_loss = 0.0
                step_num = 0

        epoch_loss /= max(1, len(loader))
        return epoch_loss

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel("Epoch")
        plt.ylabel(save_name)
        plt.savefig(join(self.args.save_path, f"{save_name}.png"))
        plt.close()

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
        wandb.finish()

    def eval(self):
        # set up model and metrics
        self.model.eval()
        self.auc.reset()
        self.f1score.reset()

        # inference loop
        for _, sample in enumerate(pbar := tqdm(self.dataloaders[1])):
            image = sample["data"].to(self.args.device, non_blocking=True)
            body_mask = (
                sample["body_mask"].to(self.args.device, non_blocking=True).float()
            )
            label = sample["label"].to(self.args.device, non_blocking=True).float()

            with torch.no_grad():
                prediction = self.model(image, body_mask)
                pred = torch.sigmoid(prediction)
                self.auc.update(pred.cpu(), label.cpu().int())
                self.f1score.update(pred.cpu(), label.cpu().int())
                pbar.set_description(f"AUC: {self.auc.compute():.4f}")
        # final metrics
        auc = self.auc.compute()
        f1 = self.f1score.compute()
        print("Final AUC:", auc)
        print("Final F1:", f1)
        self.auc.reset()
        self.f1score.reset()


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


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == "mps":
                args.device = torch.device("mps")
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def setup(rank, args):
    # initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.port}",
        world_size=args.world_size,
        rank=rank,
    )


def cleanup():
    dist.destroy_process_group()


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
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        if not args.eval:
            trainer.train()
        # Eval
        trainer.eval()


def main_worker(rank, args):
    setup(rank, args)

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
    trainer = BaseTrainer(model, dataloaders, args)

    if not args.eval:
        trainer.train()
    trainer.eval()

    cleanup()
