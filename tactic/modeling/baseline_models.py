import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


class BaselineMixer(nn.Module):
    def __init__(
        self, image_encoder, tabular_encoder, embed_dim, decoder, device, classes
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.decoder_type = decoder
        self.embed_dim = embed_dim
        self.module_device = device

        if decoder == "concat":
            self.tabular_decoder = nn.Linear(embed_dim * 2, classes)
        elif decoder == "max" or decoder == "sum":
            self.tabular_decoder = nn.Linear(embed_dim, classes)
        elif decoder == "gate":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 8),
                nn.ReLU(),
                nn.Linear(embed_dim // 8, 2),
            )
            self.tabular_decoder = nn.Linear(embed_dim, classes)
        else:
            raise ValueError("Decoder not recognized")

    def forward(self, img, tab):
        if tab is None:
            tab = torch.empty((len(img), self.embed_dim), device=self.module_device)
        else:
            tab, _ = self.tabular_encoder(*tab)
            tab = tab[:, 0, :]

        # img = img.mean(1).squeeze(1)  # no need with [CLS]
        w_img = 0.0
        if self.decoder_type == "concat":
            multi_feature = torch.cat([img, tab], dim=1)
        elif self.decoder_type == "max":
            multi_feature = torch.stack([img, tab], dim=1)
            multi_feature, _ = torch.max(multi_feature, dim=1)
        elif self.decoder_type == "sum":
            multi_feature = img + tab
        elif self.decoder_type == "gate":
            concat_features = torch.cat([tab, img], dim=1)
            gate_weight = self.gate(concat_features)
            if tab.nelement() == 0:
                gate_weight[:, 0] = float("-inf")
                # gate_weight = torch.zeros_like(gate_weight)
                # gate_weight[:, 1] = 1.0
            gate_weight = F.softmax(gate_weight, dim=1)
            w_tab = gate_weight[:, 0].unsqueeze(-1)  # (B, 1)
            w_img = gate_weight[:, 1].unsqueeze(-1)  # (B, 1)
            multi_feature = w_tab * tab + w_img * img

            # for returning
            w_img = w_img.mean().item()

        out = self.tabular_decoder(multi_feature)
        return out, w_img


class DAFT_block(nn.Module):
    def __init__(self, image_dim, tabular_dim, r=7) -> None:
        super(DAFT_block, self).__init__()
        self.dim = image_dim
        h1 = image_dim + tabular_dim
        h2 = int(h1 / r)
        self.multimodal_projection = nn.Sequential(
            nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Linear(h2, 2 * image_dim)
        )

    def forward(self, x_im, x_tab):
        num_reg_tokens = 1
        _, _, C = x_im.shape
        prefix_tokens, x = (
            x_im[:, :num_reg_tokens],
            x_im[:, num_reg_tokens:],
        )
        prefix_tokens = prefix_tokens.squeeze(1)
        x = torch.cat([prefix_tokens, x_tab], dim=1)
        attention = self.multimodal_projection(x)

        v_scale, v_shift = torch.split(attention, self.dim, dim=1)
        v_scale = v_scale.unsqueeze(1).expand(-1, -1, C)
        v_shift = v_shift.unsqueeze(1).expand(-1, -1, C)
        x = v_scale * x_im + v_shift
        return x


class DAFT(nn.Module):
    """
    Evaluation model for imaging and tabular data.
    """

    def __init__(
        self, image_encoder, tabular_encoder, vis_dim, tab_dim, device, classes
    ) -> None:
        super(DAFT, self).__init__()

        self.image_encoder = image_encoder
        self.image_encoder.eva.blocks = self.image_encoder.eva.blocks[:-1]
        self.tabular_encoder = tabular_encoder
        self.embed_dim = tab_dim
        self.module_device = device

        daft = DAFT_block(vis_dim, tab_dim)
        self.residual = image_encoder.eva.blocks[-1]
        act = nn.GELU()
        classifier = nn.Linear(vis_dim, classes)

        self.tabular_decoder = nn.ModuleList([daft, act, classifier])

        self.init_strat = "kaiming"
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module, init_gain=0.02) -> None:
        """
        Initializes weights according to desired strategy
        """
        if isinstance(m, nn.Linear):
            if self.init_strat == "normal":
                nn.init.normal_(m.weight.data, 0, 0.001)
            elif self.init_strat == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif self.init_strat == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif self.init_strat == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, img, tab) -> torch.Tensor:
        if tab is None:
            tab = torch.empty((len(img), self.embed_dim), device=self.module_device)
        else:
            tab, _ = self.tabular_encoder(*tab)
            tab = tab[:, 0, :]

        x = self.tabular_decoder[0](x_im=img, x_tab=tab)
        x = self.residual(x)
        x = self.tabular_decoder[1](x)

        # go into classifier
        if self.image_encoder.register_tokens is not None:
            num_reg_tokens = self.image_encoder.register_tokens.shape[1]
            prefix_tokens, x = (
                x[:, :num_reg_tokens],
                x[:, num_reg_tokens:],
            )
            x = self.tabular_decoder[2](prefix_tokens)
            x = x.squeeze(1, 2)  # remove token and dim dimensions
        else:
            x = self.tabular_decoder[2](x)
            x = x.mean(1)
        return x, 0.0


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=False):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)
        # print(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim, k_dim * 2, bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim, k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, q, k, visualize=False):
        B, N_k, K = k.shape
        _, N_q, _ = q.shape
        kv = (
            self.kv_proj(k)
            .reshape(B, N_k, 2, self.num_heads, K // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  #
        k, v = kv[0], kv[1]  # (B,H,N,C)
        q = (
            self.q_proj(q)
            .reshape(B, N_q, self.num_heads, K // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B,H,N,C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, K)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TIP_block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        is_cross_attention=False,
        encoder_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if self.is_cross_attention:
            self.cross_attn = CrossAttention(
                q_dim=dim,
                k_dim=encoder_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.cross_norm = norm_layer(dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, encoder_hidden_states=None, mask=None):
        tmp = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(tmp)
        if self.is_cross_attention:
            assert encoder_hidden_states is not None
            tmp = self.cross_attn(self.cross_norm(x), encoder_hidden_states)
            x = x + self.drop_path(tmp)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TIP(nn.Module):
    """
    Tabular Transformer Encoder based on BERT
    """

    def __init__(
        self, image_encoder, tabular_encoder, vis_dim, tab_dim, device, classes
    ) -> None:
        super(TIP, self).__init__()
        multimodal_dim = 512  # TIP
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.module_device = device
        self.embed_dim = tab_dim

        image_proj = nn.Linear(vis_dim, multimodal_dim)
        tabular_proj = (
            nn.Linear(tab_dim, multimodal_dim)
            if tab_dim != multimodal_dim
            else nn.Identity()
        )
        transformer_blocks = nn.ModuleList(
            [
                TIP_block(
                    dim=multimodal_dim,
                    is_cross_attention=True,
                    encoder_dim=multimodal_dim,
                )
                for i in range(4)
            ]
        )
        norm = nn.LayerNorm(multimodal_dim)

        classifier = nn.Linear(multimodal_dim, classes)

        self.tabular_decoder = nn.ModuleList(
            [image_proj, tabular_proj, transformer_blocks, norm, classifier]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def forward(self, img, tab):
        if tab is None:
            tab = torch.empty((len(img), 1, self.embed_dim), device=self.module_device)
        else:
            tab, _ = self.tabular_encoder(*tab)

        img = self.tabular_decoder[0](img)
        tab = self.tabular_decoder[1](tab)

        for i, transformer_block in enumerate(self.tabular_decoder[2]):
            x = transformer_block(tab, encoder_hidden_states=img)
        x = self.tabular_decoder[3](x)
        x = self.tabular_decoder[4](x[:, 0, :])
        return x, 0.0
