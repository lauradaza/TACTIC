# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FusionHead(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        tabular_encoder,
        tabular_decoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.tabular_decoder = tabular_decoder

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = torch.stack([x["scan"] for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "attributes" in image_record:
                attributes = image_record["attributes"]
            else:
                attributes = None
            sparse_embeddings = self.tabular_encoder(attributes)
            predictions = self.tabular_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                sparse_prompt_embeddings=sparse_embeddings,
            )
            outputs.append(
                {
                    "predictions": predictions,
                }
            )
        return outputs
