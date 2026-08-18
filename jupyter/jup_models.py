import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactic.modeling.mask_decoder import MLP

from tarte_ai import TARTE_Base


class TransformerEncoderLayer_attn(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.saved_attn = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        self.saved_attn = attn.detach()
        return self.dropout1(x)


class TARTE_Base_attn(TARTE_Base):
    def __init__(
        self,
        dim_input: int,
        dim_transformer: int,
        dim_feedforward: int,
        num_heads: int,
        num_layers_transformer: int,
        dropout: float,
    ):
        super().__init__(
            dim_input=dim_input,
            dim_transformer=dim_transformer,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            dropout=dropout,
        )

        encoder_layer = TransformerEncoderLayer_attn(
            d_model=dim_transformer,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=True,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_transformer,
            enable_nested_tensor=False,
        )


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.save_attention = False

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        attn = attn.softmax(dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn.detach()


class TwoWayAttentionBlock1D(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
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

    def forward(self, queries, keys, query_skip, key_skip):
        # Self attention block (tokens)
        if self.skip_first_layer_pe:
            queries, attn_matrix = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_skip
            attn_out, attn_matrix = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_skip
        k = keys + key_skip
        attn_out, _ = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_skip
        k = keys + key_skip
        attn_out, _ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys, attn_matrix


class TwoWayTransformer1D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
    ) -> None:
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
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, point_embedding):
        if len(image_embedding.shape) == 2:
            image_embedding = image_embedding[:, None]

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        attn_matrices = []
        for layer in self.layers:
            queries, keys, attn_matrix = layer(
                queries=queries,
                keys=keys,
                query_skip=point_embedding,
                key_skip=image_embedding,
            )
            attn_matrices.append(attn_matrix)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_embedding
        attn_out, _ = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys, attn_matrices


def self_attention_rollout(attn_matrices, device):
    # get attention shape from first layer
    A0 = attn_matrices[0]
    B, H, T, _ = A0.shape

    # initialize rollout as identity
    rollout = torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)

    for attn_m in attn_matrices:
        A = attn_m  # (B, H, T, T)
        A = A.mean(dim=1)  # avg heads → (B, T, T)

        # add residual
        A = A + torch.eye(T, device=device)

        # row-normalize
        A = A / A.sum(dim=-1, keepdim=True)

        rollout = A @ rollout

    return rollout


class TabularDecoder(nn.Module):

    def __init__(
        self,
        *,
        transformer_dim: int,
        cad_head_depth: int = 3,
        cad_head_hidden_dim: int = 256,
        init_embeddings=None,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer1D(
            depth=2,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        self.out_tokens = nn.Embedding.from_pretrained(init_embeddings, freeze=True)
        self.num_out_tokens = self.out_tokens.weight.shape[0]
        self.cad_prediction_head = MLP(
            transformer_dim, cad_head_hidden_dim, 1, cad_head_depth
        )

    def forward(
        self,
        image_embeddings,
        sparse_prompt_embeddings,
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = self.out_tokens.weight.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # src = image_embeddings

        # Run the transformer
        hs, src, attn_matrices = self.transformer(image_embeddings, tokens)
        # final_attn_matrix = self_attention_rollout(attn_matrices, hs.device)
        final_attn_matrix = attn_matrices[-1].mean(1)
        out_tokens_post = hs[:, : self.num_out_tokens, :]

        # Generate mask quality predictions
        cad_pred = self.cad_prediction_head(out_tokens_post)

        return cad_pred, final_attn_matrix
