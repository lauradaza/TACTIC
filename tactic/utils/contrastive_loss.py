import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainLossMultiSequence(nn.Module):
    """
    Batch order must be:
      [fat0, water0, fat1, water1, ..., fatN, waterN]

    Model outputs:
      recon:      [B, 1, H, W, D]
      tokens:     [B, K, C]
      mae_mask:   [B, 1, H, W, D]
      keep_idx:   [B, K]
      mask_patch: [B, L]   (True for masked patch, optional but preferred)

    Extra input:
      valid_voxel_mask_patient: [B//2, 1, H, W, D]
    """

    def __init__(self):
        super().__init__()

    def reconstruction_loss(self, recon, target, mae_mask, valid_voxel_mask):
        use = mae_mask.bool() & valid_voxel_mask.bool()
        if use.sum() == 0:
            return target.new_zeros(())
        per_voxel = F.mse_loss(recon, target, reduction="none")
        return per_voxel[use].mean()

    def assert_token_masks(self, in_mask, name="mask_patch"):
        _mask_patch0 = in_mask[0::2]
        _mask_patch1 = in_mask[1::2]
        assert torch.all(
            _mask_patch0 == _mask_patch1
        ), f"The {name} for the different sequences should be identical"
        return _mask_patch0

    def forward(
        self,
        target,
        recon,
        valid_voxel_mask_patient,
        mae_mask,
    ):
        B = recon.shape[0]
        if B % 2 != 0:
            raise ValueError(
                "Batch size must be even for [fat0, water0, fat1, water1, ...] ordering."
            )

        # Reconstruction loss on masked tokens within the body
        valid_voxel_mask = valid_voxel_mask_patient.repeat_interleave(2, dim=0)
        loss_rec = self.reconstruction_loss(
            recon=recon,
            target=target,
            mae_mask=~mae_mask,  # we need to invert it so 0=kept, 1=masked
            valid_voxel_mask=valid_voxel_mask,  # 0=bg, 1=body
        )

        return {"loss": loss_rec}
