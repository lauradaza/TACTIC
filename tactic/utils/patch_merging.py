import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactic.modeling.primus import LayerNormNd


def masked_max_pool3d(x, mask, merge_factor=(2, 2, 2)):
    mask = mask.bool()

    pooled_mask = F.max_pool3d(mask.float(), merge_factor, merge_factor) > 0

    # invalid voxels → -inf
    x_masked = x.masked_fill(~mask, float("-inf"))  # TODO
    pooled = F.max_pool3d(x_masked, merge_factor, merge_factor)

    pooled = pooled.masked_fill(~pooled_mask, 0)
    return pooled


def masked_avg_pool3d_conv(x, mask, merge_factor=(2, 2, 2), eps=1e-5):
    B, C, _, _, _ = x.shape
    fd, fh, fw = merge_factor

    mask = mask.float()
    mask_c = mask.expand(-1, C, -1, -1, -1)

    kernel = torch.ones(C, 1, fd, fh, fw, device=x.device)

    value_sum = F.conv3d(x * mask_c, kernel, stride=merge_factor, groups=C)
    mask_sum = F.conv3d(mask_c, kernel, stride=merge_factor, groups=C)

    return value_sum / mask_sum.clamp(min=eps)


class MaskedAvgMaxPool3D(nn.Module):
    def __init__(
        self,
        in_ch=2,
        embed_dim=864,
        subpatch_size=(4, 4, 4),
        merge_factor=(4, 4, 4),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.subpatch_size = subpatch_size
        self.merge_factor = merge_factor

        # 1) Linear merging layer
        self.merge = nn.Conv3d(
            in_ch * 2,  # concat
            in_ch,
            kernel_size=1,
        )
        # self.norm_after_merge = LayerNormNd(in_ch)

        # 2) Subpatch embedding
        self.proj = nn.Conv3d(
            in_ch, embed_dim, kernel_size=subpatch_size, stride=subpatch_size
        )
        self.norm_after_tokens = LayerNormNd(embed_dim)

    def forward(self, x, mask):
        if max(self.merge_factor) == 1:
            x = self.proj(x)
            x = self.norm_after_tokens(x)
            return x, x, mask

        mask_dup = mask
        if mask.shape[0] == x.shape[0] // 2:
            mask_dup = mask.repeat_interleave(2, dim=0)

        # x_in = x.clone().detach().cpu().numpy()
        # affine = np.eye(4)
        # nib.save(nib.Nifti1Image(x_in[0,0], affine), "input.nii.gz")
        # nib.save(nib.Nifti1Image(x[0,0].cpu().detach().float().numpy(), affine), "small.nii.gz")
        avg = masked_avg_pool3d_conv(x, mask_dup, self.merge_factor)
        mx = masked_max_pool3d(x, mask_dup, self.merge_factor)

        pooled = torch.cat([avg, mx], dim=1)

        pooled = self.merge(pooled)
        mask = F.interpolate(mask, size=pooled.shape[-3:], mode="nearest")
        # x = self.norm_after_merge(x)

        x = self.proj(pooled)
        x = self.norm_after_tokens(x)
        return x, pooled, mask


class PatchEmbedMerge3D(nn.Module):
    """
    Subpatch embedding + PatchMerging in one module.
    Input: [B, C_in, D, H, W] (whole-body MRI)
    Output: [B, C_merged, D', H', W'] (tokens for transformer)
    """

    def __init__(
        self,
        in_ch=2,
        embed_dim=864,
        subpatch_size=(16, 16, 8),
        merge_factor=(2, 2, 2),
    ):
        super().__init__()
        self.subpatch_size = subpatch_size
        self.merge_factor = merge_factor
        self.embed_dim = embed_dim // np.prod(merge_factor)
        self.C_merged = embed_dim  # can also increase if desired

        # 1) Subpatch embedding
        self.proj = nn.Conv3d(
            in_ch, embed_dim, kernel_size=subpatch_size, stride=subpatch_size
        )
        self.norm_before_merge = nn.LayerNorm(embed_dim)

        # 2) Linear merging layer
        self.linear_merge = nn.Linear(
            embed_dim * merge_factor[0] * merge_factor[1] * merge_factor[2],
            self.C_merged,
        )
        self.norm_after_merge = nn.LayerNorm(self.C_merged)

    def forward(self, x, mask=None):
        B, C_in, D, H, W = x.shape
        fd, fh, fw = self.merge_factor

        # --- Subpatch embedding ---
        x = self.proj(x)  # [B, C_sub, D_sub, H_sub, W_sub]
        B, C_sub, D_sub, H_sub, W_sub = x.shape

        # LayerNorm across channels
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D_sub, H_sub, W_sub, C_sub]
        x = self.norm_before_merge(x)
        x = x.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # back to [B, C_sub, D_sub, H_sub, W_sub]

        # --- Patch merging ---
        # Ensure divisible by merge factor
        assert (
            D_sub % fd == 0 and H_sub % fh == 0 and W_sub % fw == 0
        ), "Subpatch dims must be divisible by merge factor"

        # Group subpatches
        x = x.view(B, C_sub, D_sub // fd, fd, H_sub // fh, fh, W_sub // fw, fw)
        x = x.permute(
            0, 2, 4, 6, 3, 5, 7, 1
        ).contiguous()  # [B, D', H', W', fd, fh, fw, C_sub]
        x = x.view(
            B, D_sub // fd, H_sub // fh, W_sub // fw, -1
        )  # flatten subpatches: [B, D', H', W', fd*fh*fw*C_sub]

        # Linear projection
        x = self.linear_merge(x)  # [B, D', H', W', C_merged]

        x = self.norm_after_merge(x)

        # Final shape: [B, C_merged, D', H', W']
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x, None, mask
