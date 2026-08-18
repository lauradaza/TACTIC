# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from tactic.modeling.mask_decoder import MLP
from tactic.modeling.mask_decoder3D import Attention


class TwoWayTransformer1D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
        attn_scale=1.0,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock1D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    attn_scale=attn_scale,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, point_embedding):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        if len(image_embedding.shape) == 2:
            image_embedding = image_embedding[:, None]

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_skip=point_embedding,
                key_skip=image_embedding,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_embedding
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock1D(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        attn_scale=1.0,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe
        self.attn_scale = attn_scale

    def forward(self, queries, keys, query_skip, key_skip):
        # Self attention block (tokens)
        scale = min(1.0, max(self.attn_scale, 1 - queries.shape[1] / 233))
        # scale = self.attn_scale
        if self.skip_first_layer_pe:
            queries = self.self_attn(
                q=queries,
                k=queries,
                v=queries,
                attn_scale=scale,
            )
        else:
            q = queries + query_skip
            attn_out = self.self_attn(q=q, k=q, v=queries, attn_scale=scale)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_skip
        k = keys + key_skip
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_skip
        k = keys + key_skip
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class FusionDecoder(nn.Module):

    def __init__(
        self,
        *,
        transformer_dim: int,
        cad_head_depth: int = 3,
        cad_head_hidden_dim: int = 256,
        init_embeddings=None,
        attn_scale=1.0,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer1D(
            depth=2,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
            attn_scale=attn_scale,
        )

        self.out_tokens = nn.Embedding.from_pretrained(init_embeddings, freeze=True)
        self.num_out_tokens = self.out_tokens.weight.shape[0]
        self.prediction_head = MLP(
            transformer_dim, cad_head_hidden_dim, 1, cad_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Concatenate output tokens
        output_tokens = self.out_tokens.weight.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # src = image_embeddings

        # Run the transformer
        hs, src = self.transformer(image_embeddings, tokens)
        out_tokens_post = hs[:, : self.num_out_tokens, :]

        # Generate mask quality predictions [B, num_out_tokens, 1]
        pred = self.prediction_head(out_tokens_post)

        return pred.squeeze(1)
