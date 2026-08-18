import os
import torch
import torch.nn as nn

from tactic.nnunet.eva_class import EvaClass
from tactic.modeling.tabular_encoder import TabularClassifier
from tactic.modeling.tabular_sam import SamTabular

from jup_models import TabularDecoder

from tarte_ai import TARTE_TablePreprocessor
from tactic.dataloading.imTab_dataset import WBImgTabDataset, WBImgTabCollator

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    ClipIntensityPercentilesd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    EnsureTyped,
)


class TabularFeatures(TabularClassifier):
    def __init__(self, num_classes, embed_dim, pretrained, device):
        super().__init__(num_classes, "", save_attn=True)
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
    print("Building the models...")
    input_image_size = args.img_size
    patch_embed_size = args.patch_size

    vis_embed_size = 864
    tabular_embed_dim = 768

    image_encoder = EvaClass(
        input_channels=2,
        embed_dim=vis_embed_size,
        patch_embed_size=patch_embed_size,
        output_channels=args.num_classes,
        num_register_tokens=1,  # class token
        eva_depth=16,
        eva_numheads=12,
        input_shape=input_image_size,
        drop_path_rate=0.2,
        scale_attn_inner=True,
        init_values=0.1,
        patch_drop_rate=0.4,
        merge_tokens=args.merge_tokens,
    ).to(args.device)

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
        embed_dim=tabular_embed_dim,
        pretrained=args.pretrained_tab,
        device=args.device,
    )

    # Use the name transformation and initial mapping from TARTE
    processor = TARTE_TablePreprocessor()
    processor._load_lm_model()
    out_disease = (
        args.disease
        if args.disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )
    class_names = ["Not presence " + out_disease, out_disease + " diagnosed"]

    embb_names = torch.tensor(processor._transform_names(class_names))
    embb = tabular_encoder.tabular_encoder.initial_e(embb_names)

    # finally SAM
    sam_model = SamTabular(
        image_encoder=image_encoder,
        tabular_encoder=tabular_encoder,
        tabular_decoder=TabularDecoder(
            transformer_dim=tabular_embed_dim,
            cad_head_depth=3,
            cad_head_hidden_dim=256,
            init_embeddings=embb,
        ),
    ).to(args.device)

    model_root = "/ictstr01/groups/iml/projects/laura.daza/miccai26/workshop"
    ckp_path = os.path.join(model_root, args.task_name, "model_latest.pth")
    last_ckpt = torch.load(ckp_path, map_location=args.device, weights_only=False)

    sam_model.load_state_dict(last_ckpt["model_state_dict"])
    print("SAM weights from:", ckp_path, "- epoch", last_ckpt["epoch"])
    return sam_model


def get_dataloaders(args):
    print("Building the dataloaders...")
    test_transform = Compose(
        [
            EnsureChannelFirstd(keys=["img", "mask"], channel_dim=0),  # [1, D, H, W]
            ClipIntensityPercentilesd(
                keys=["img"], lower=1, upper=99
            ),  # only clip lower tail
            NormalizeIntensityd(
                keys=["img"],
                # nonzero=True,  # False: use full volume
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

    label_key = (
        args.disease.capitalize().replace("_", " ")
        if args.disease != "copd"
        else "Chronic Obstructive Pulmonary Disease"
    )

    train_data = WBImgTabDataset(
        root_dir="/path/to/img/data",
        csv_path="/path/to/csv_files/dir/",
        transform=test_transform,
        split="balanced train",
        label_key=label_key,
        disease=args.disease,
    )

    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_prepper.fit(train_data.tabular_data)
    collator = WBImgTabCollator(tarte_tab_prepper)

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    return train_dataloader
