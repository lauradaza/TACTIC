import os
import json

import torch
import torch.nn as nn

from tarte_ai import TARTE_Base
from huggingface_hub import hf_hub_download

from jup_models import TARTE_Base_attn


## TARTE - Base Model
class TARTE_embed(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_transformer: int,
    ):
        super(TARTE_embed, self).__init__()

        # Initial linear layers for cell features
        self.initial_x = nn.Sequential(
            nn.Linear(dim_input, dim_transformer),
            nn.ReLU(),
            nn.LayerNorm(dim_transformer),
        )

        # Initial linear layers for column features
        self.initial_e = nn.Sequential(
            nn.Linear(dim_input, dim_transformer),
            nn.ReLU(),
            nn.LayerNorm(dim_transformer),
        )

        self.norm = nn.LayerNorm(dim_transformer)

    def forward(self, x, edge_attr, mask=None):

        # Initial linear layers for cell and column features
        x = self.initial_x(x)
        edge_attr = self.initial_e(edge_attr)

        # Combine the cell and column features with addition
        z = x + edge_attr
        return self.norm(z)


class TabularClassifier(nn.Module):
    def __init__(
        self, num_classes, tabular_checkpoint, just_embed=False, save_attn=False
    ):
        super().__init__()
        self.save_attn = save_attn
        self.tabular_checkpoint = tabular_checkpoint
        self.just_embed = just_embed
        self.initialize_tabular_model()
        self.tabular_head = torch.nn.Sequential(torch.nn.Linear(768, num_classes))

    def initialize_tabular_model(self):
        self.pretrain_configs, weights = self.load_tarte_config()
        if self.just_embed:
            self.tabular_encoder = TARTE_embed(
                dim_input=self.pretrain_configs["dim_input"],
                dim_transformer=self.pretrain_configs["dim_transformer"],
            )
        else:
            if not self.save_attn:
                self.tabular_encoder = TARTE_Base(
                    dim_input=self.pretrain_configs["dim_input"],
                    dim_transformer=self.pretrain_configs["dim_transformer"],
                    dim_feedforward=self.pretrain_configs["dim_feedforward"],
                    num_heads=self.pretrain_configs["num_heads"],
                    num_layers_transformer=self.pretrain_configs[
                        "num_layers_transformer"
                    ],
                    dropout=self.pretrain_configs["dropout"],
                )
            else:
                self.tabular_encoder = TARTE_Base_attn(
                    dim_input=self.pretrain_configs["dim_input"],
                    dim_transformer=self.pretrain_configs["dim_transformer"],
                    dim_feedforward=self.pretrain_configs["dim_feedforward"],
                    num_heads=self.pretrain_configs["num_heads"],
                    num_layers_transformer=self.pretrain_configs[
                        "num_layers_transformer"
                    ],
                    dropout=self.pretrain_configs["dropout"],
                )
                print("TARTE_base saving attn matrices")
        if len(self.tabular_checkpoint) != 0:
            print("LOADED TABULAR ENCODER")
            check = torch.load(self.tabular_checkpoint)
            check = {
                k.replace("tabular_encoder.tarte_base.", ""): v
                for k, v in check.items()
            }
            missing_keys, un = self.tabular_encoder.load_state_dict(check, strict=False)
            if missing_keys:
                print("missing keys tabular encoder: ", missing_keys)
        else:
            print("TARTE initialized from original weights")
            pretrain_weights = {
                k.replace("tarte_base.", ""): v for k, v in weights.items()
            }
            missing, _ = self.tabular_encoder.load_state_dict(
                pretrain_weights, strict=False
            )
            if missing:
                print("missing keys tabular encoder: ", missing)

    def load_tarte_config(self, device="cpu"):
        base_path = "/lustre/groups/iml/projects/marta/"
        cache_dir = os.path.join(base_path, "data/pretrained_weights")

        repo_id = "inria-soda/tarte"

        # Load weights
        weights_file = "tarte_pretrained_weights.pt"
        model_path = hf_hub_download(
            repo_id=repo_id, filename=weights_file, cache_dir=cache_dir
        )
        pretrain_weights = torch.load(
            model_path, map_location=device, weights_only=True
        )

        # Load configs
        config_file = "tarte_pretrained_configs.json"
        config_path = hf_hub_download(
            repo_id=repo_id, filename=config_file, cache_dir=cache_dir
        )
        with open(config_path) as f:
            pretrain_model_configs = json.load(f)

        return pretrain_model_configs, pretrain_weights

    def forward(self, x, edge_attr, mask):
        tabular_features = self.tabular_encoder(x, edge_attr, mask)
        logits = self.tabular_head(tabular_features[:, 0, :])
        return tabular_features, logits
