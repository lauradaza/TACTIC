from typing import Tuple
import torch
from torch import nn
import numpy as np
from timm.layers import RotaryEmbeddingCat
from einops import rearrange
from timm.layers import Mlp

from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from tactic.modeling.primus import PatchEmbed
from tactic.modeling.eva_transformer import MaskedEva
from tactic.modeling.evaMAE import PatchEmbedMerge3D, MaskedAvgMaxPool3D

from tactic.utils.body_masking import BodyPatchDropout


class EvaClass(nn.Module):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        output_channels: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        drop_path_scale: bool = True,
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
        merge_tokens=False,
        full_output=False,
    ):
        """
        consists of a UNet encoder, a EVA ViT bottleneck and a UNet decoder
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3d is supported"

        super().__init__()

        self.patch_embed_size = patch_embed_size
        if isinstance(merge_tokens, list) or isinstance(merge_tokens, tuple):
            self.down_projection = MaskedAvgMaxPool3D(
                input_channels,
                embed_dim=embed_dim,
                subpatch_size=patch_embed_size,
                merge_factor=merge_tokens,
            )
            self.patch_embed_size = [
                mg * ds for mg, ds in zip(merge_tokens, patch_embed_size)
            ]
            print("Using masked pooling (new)")
        elif merge_tokens is True:
            sub_token = [i // 2 for i in patch_embed_size]
            self.down_projection = PatchEmbedMerge3D(
                input_channels, embed_dim=embed_dim, subpatch_size=sub_token
            )
            print("Using patch merging (old)")
        else:
            self.down_projection = PatchEmbed(
                patch_embed_size, input_channels, embed_dim
            )

        assert all([j % i == 0 for i, j in zip(self.patch_embed_size, input_shape)])
        token_shape = [i // ds for i, ds in zip(input_shape, self.patch_embed_size)]
        num_tokens = np.prod(token_shape)
        print(f"Total number of tokens: {num_tokens}")
        # we need to compute the ref_feat_shape for eva
        self.eva = MaskedEva(
            token_shape,
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple(token_shape),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            patch_drop_module=BodyPatchDropout,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
            patch_embed_size=patch_embed_size,
        )

        self.classifier = Mlp(embed_dim, embed_dim, output_channels)

        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
                if num_register_tokens
                else None
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
            print(f"Using {num_register_tokens} register tokens")
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.classifier.apply(InitWeights_He(1e-2))

        self.full_output = full_output
        # eva has its own initialization

    def forward(self, x, mask=None):
        x, _, mask = self.down_projection(x, mask)

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, _ = self.eva(x, mask)

        if self.full_output:
            return self.classifier(x)

        # go into classifier
        if self.register_tokens is not None:
            num_reg_tokens = self.register_tokens.shape[1]
            prefix_tokens, x = (
                x[:, :num_reg_tokens],
                x[:, num_reg_tokens:],
            )
            x = self.classifier(prefix_tokens)
            x = x.squeeze(1, 2)  # remove token and dim dimensions
        else:
            x = self.classifier(x)
            x = x.mean(1)

        return x
