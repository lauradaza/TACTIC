from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def voxel_mask_to_patch_mask(body_mask, patch_size, body_threshold=0.5):
    """
    body_mask: (B, 1, H, W, Z) bool
    patch_size: (ph, pw, pz)

    returns: (B, P) bool, True = body patch
    """
    B, _, H, W, Z = body_mask.shape
    ph, pw, pz = patch_size

    mask = body_mask.view(
        B,
        H // ph,
        ph,
        W // pw,
        pw,
        Z // pz,
        pz,
    )

    # fraction of body voxels per patch
    body_fraction = mask.float().mean(dim=(2, 4, 6))

    # patch is body if ≥ threshold
    patch_mask = body_fraction >= body_threshold
    return patch_mask.flatten(1)


class BodyPatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    return_indices: torch.jit.Final[bool]

    def __init__(
        self,
        token_shape: Tuple[int, int, int],
        prob: float = 0.5,
        num_prefix_tokens: int = 0,
        return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.token_shape = token_shape
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens
        self.return_indices = return_indices

    def forward(
        self, x, mask
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = (
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens :],
            )
        else:
            prefix_tokens = None

        Bx = x.shape[0]
        B = mask.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1.0 - self.prob)))

        patched_mask = voxel_mask_to_patch_mask(mask, self.token_shape)
        patched_mask = -(~patched_mask * 0.7) + 1  # [0.3, 1] => [bg, body]

        # keep the largests random indices (bg * 0.3 -> keep it small)
        keep_indices = torch.argsort(
            torch.rand(B, L, device=x.device) * patched_mask, dim=-1
        )[:, -num_keep:]

        if B == Bx // 2:
            keep_indices = keep_indices.repeat_interleave(2, dim=0)

        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x
