import numpy as np
from typing import Tuple

import torch
from torch import nn
from tactic.modeling.eva_transformer import MaskedEva, Eva
from tactic.utils.body_masking import BodyPatchDropout
from tactic.modeling.primus import PatchEmbed, PatchDecode, LayerNormNd
from tactic.utils.patch_merging import (
    PatchEmbedMerge3D,
    MaskedAvgMaxPool3D,
)

from einops import rearrange
from timm.layers import RotaryEmbeddingCat


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class EncoderProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, intermediate_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
        )
        self.last_layer = nn.Linear(intermediate_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.projection(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class EvaMAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        output_channels: int,
        encoder_eva_depth: int = 24,
        encoder_eva_numheads: int = 16,
        decoder_eva_depth: int = 24,
        decoder_eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_emb: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,
        drop_path_scale: bool = True,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        do_up_projection=True,
        init_values=None,
        scale_attn_inner=False,
        merge_tokens=(2, 2, 2),
        patch_drop_module=BodyPatchDropout,
    ):
        """
        Masked Autoencoder with EVA attention-based encoder and decoder.
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.patch_embed_size = [
            mg * ds for mg, ds in zip(merge_tokens, patch_embed_size)
        ]
        self.embed_dim = embed_dim

        # Patch embedding for encoder
        if merge_tokens is not None:
            # sub_token = [i // j for i, j in zip(patch_embed_size, merge_tokens)]
            # self.down_projection = PatchEmbedMerge3D(
            #     input_channels,
            #     embed_dim=embed_dim,
            #     subpatch_size=sub_token,
            #     merge_factor=merge_tokens,
            # )
            self.down_projection = MaskedAvgMaxPool3D(
                input_channels,
                embed_dim=embed_dim,
                subpatch_size=patch_embed_size,  # sub_token
                merge_factor=merge_tokens,
            )
        else:
            self.down_projection = PatchEmbed(
                patch_embed_size, input_channels, embed_dim
            )

        # Encoder using EVA
        # token_shape = [i // ds for i, ds in zip(input_shape, patch_embed_size)]
        token_shape = [i // ds for i, ds in zip(input_shape, self.patch_embed_size)]
        self.eva = MaskedEva(
            token_shape,
            embed_dim=embed_dim,
            depth=encoder_eva_depth,
            num_heads=encoder_eva_numheads,
            ref_feat_shape=tuple(token_shape),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
            patch_drop_module=patch_drop_module,
            patch_embed_size=patch_embed_size,
        )

        # Patch embedding for decoder
        if do_up_projection:
            self.up_projection = PatchDecode(
                patch_embed_size,
                embed_dim,
                output_channels,
                norm=decoder_norm,
                activation=decoder_act,
            )
        else:
            self.up_projection = nn.Identity()

        # Projection for contrastive learning
        self.enc_projection = EncoderProjection(
            input_dim=embed_dim, output_dim=embed_dim
        )

        # Decoder using EVA
        if decoder_eva_depth > 0:
            self.decoder = Eva(
                embed_dim=embed_dim,
                depth=decoder_eva_depth,  # eva_depth,
                num_heads=decoder_eva_numheads,  # eva_numheads,
                ref_feat_shape=tuple(token_shape),
                num_reg_tokens=num_register_tokens,
                use_rot_pos_emb=use_rot_pos_emb,
                use_abs_pos_emb=use_abs_pos_emb,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                patch_drop_rate=0,  # No drop in the decoder
                proj_drop_rate=proj_drop_rate,
                attn_drop_rate=attn_drop_rate,
                rope_impl=rope_impl,
                rope_kwargs=rope_kwargs,
                init_values=init_values,
                scale_attn_inner=scale_attn_inner,
            )
            self.use_decoder = True
        else:
            self.use_decoder = False
            # self.decoder = DecoderIdentity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

        self.down_projection.apply(InitWeights_He(1e-2))

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(
            B, num_masked, 1
        )  # Shape: (B, num_masked, C)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)
        restored_mask = torch.ones(B, num_patches, dtype=torch.bool, device=device)

        # Create a flat indices tensor for assignment
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)  # Shape: (B, 1)

        # Flatten keep_indices to use for scatter
        flat_indices = torch.arange(num_patches, device=device).repeat(
            B, 1
        )  # Shape: (B, num_patches)
        mask = torch.zeros_like(
            flat_indices, dtype=torch.bool, device=device
        )  # Mask for "kept" indices

        # Mark kept positions
        mask[batch_indices, keep_indices] = True

        # Assign kept positions
        restored[batch_indices, keep_indices] = x

        # Assign mask tokens to the remaining positions
        masked_positions = torch.where(~mask)
        restored[masked_positions] = mask_tokens.view(-1, C)
        restored_mask[masked_positions] = False
        return restored, restored_mask

    def forward(self, x, mask=None):
        # Encode patches
        x, pooled, mask = self.down_projection(x, mask)
        FW, FH, FD = pooled.shape[2:]
        B, C, W, H, D = x.shape
        x = rearrange(x, "b c w h d -> b (w h d) c")

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x, mask=mask)
        # Restore full sequence with mask tokens
        num_patches = W * H * D
        restored_x, restoration_mask = self.restore_full_sequence(
            encoded, keep_indices, num_patches
        )

        # Project encoded features for contrastive learning
        encoded = self.enc_projection(encoded)

        # Expand restoration mask to full spatial dimensions
        restoration_mask = rearrange(
            restoration_mask, "b (w h d) -> b w h d", w=W, h=H, d=D
        )
        mae_mask = (
            restoration_mask.repeat_interleave(FW // W, dim=1)
            .repeat_interleave(FH // H, dim=2)
            .repeat_interleave(FD // D, dim=3)
        )
        mae_mask = mae_mask[:, None, ...]  # Add channel dimension  # [B, 1, W, H, D]

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        # Project back to output shape
        decoded = rearrange(decoded, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        decoded = self.up_projection(decoded)

        output = {
            "pooled": pooled.detach(),
            "rec": decoded,
            "body_mask": mask,
            "features": encoded,
            "mae_mask": mae_mask,
            "keep_indices": keep_indices,
            "res_mask": restoration_mask,
        }

        return output


if __name__ == "__main__":
    # Toy example for testing
    input_shape = (64, 64, 64)
    patch_embed_size = (8, 8, 8)
    model = EvaMAE(
        input_channels=3,
        embed_dim=192,
        patch_embed_size=patch_embed_size,
        output_channels=3,
        input_shape=input_shape,
        eva_depth=6,
        eva_numheads=8,
    )

    # Random input tensor
    x = torch.rand((2, 3, *input_shape))  # Batch size 2

    # Forward pass
    output, keep_indices = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
