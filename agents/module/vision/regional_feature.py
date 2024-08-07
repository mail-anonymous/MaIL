import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from agents.models.oc_ddpm.module import LayerNorm


class RoIAlignWrapper(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True,
                 bbox_size=4):
        super().__init__()
        assert (aligned == True)
        self.output_size = output_size
        self.bbox_size = bbox_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned=aligned)  # [K, C, output_size[0], output_size[1]]  k is the number of bboxes

    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_size = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = out.reshape(batch_size, bbox_size, channel_size, *self.output_size)
        return out

    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (input_shape[0], self.output_size[0], self.output_size[1])


class FlattenProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim):
        super().__init__()

        assert (len(input_shape) == 4), "You should not use FlattenProjection if not having a tensor with 4 dimensions (excluding batch dimension)"  # B C H W
        in_dim = input_shape[-3] * input_shape[-2] * input_shape[-1]  # C*H*W
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=2)  # (B,C,H,W) --> (B,C,H*W)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class RegionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x, obj_att=False):
        (
            B,
            T,
            A,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if obj_att:
            # B, T, A, C

            k = (self.key(x).view(B, T, A, self.n_head, C // self.n_head).transpose(2, 3)
                 )  # (B, T, nh, A, hs)
            q = (
                self.query(x).view(B, T, A, self.n_head, C // self.n_head).transpose(2, 3)
            )  # (B, T, nh, A, hs)
            v = (
                self.value(x).view(B, T, A, self.n_head, C // self.n_head).transpose(2, 3)
            )  # (B, T, nh, A, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.mask[:, :, :, :A, :A] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)

            # obj att: (B, T, nh, A, A) x (B, T, nh, A, hs) -> (B, T, nh, A, hs)
            y = att @ v
            y = (
                y.transpose(2, 3).transpose(1, 2).contiguous().view(B, T, A, C)
            )  # re-assemble all head outputs side by side
        else:
            # B, A, T, C || temporal attention
            x = x.permute(0, 2, 1, 3)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = (self.key(x).view(B, A, T, self.n_head, C // self.n_head).transpose(2, 3)
                 )  # (B, A, nh, T, hs)
            q = (
                self.query(x).view(B, A, T, self.n_head, C // self.n_head).transpose(2, 3)
            )  # (B, A, nh, T, hs)
            v = (
                self.value(x).view(B, A, T, self.n_head, C // self.n_head).transpose(2, 3)
            )  # (B, A, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.mask[:, :, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            # time att: (B, A, nh, T, T) x (B, A, nh, T, hs) -> (B, A, nh, T, hs)
            y = att @ v

            y = (
                y.transpose(2, 3).contiguous().view(B, T, A, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class RegionalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.1, attn_drop=0.1):
        super(RegionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0

        self.dim_per_head = embed_dim // num_heads

        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        self.att_drop = nn.Dropout(attn_drop)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        B, T, A, C = x.size()
        x = x.view(B*T, A, C)

        if context.dim() == 3:
            context = context.view(B*T, -1)
            context = context.unsqueeze(1).expand(B*T, A, -1)
        elif context.dim() == 2:
            context = context.unsqueeze(1).expand(B*T, A, -1)

        q = self.query_layer(x).view(B*T, A, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)  # --> BT nh A dh
        k = self.key_layer(context).view(B*T, A, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        v = self.value_layer(context).view(B*T, A,  self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.dim_per_head)  # --> BT nh A A
        attention = F.softmax(attention, dim=-1)
        attention = self.att_drop(attention)

        out = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous().view(B*T, A, C)
        out = self.dropout(out).view(B, T, A, C)

        return out

    def get_params(self):
        return self.parameters()


class AttentionBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
            # obj_att: bool
    ):
        super().__init__()

        # self.obj_att = obj_att

        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.ln3 = LayerNorm(n_embd, bias=False)
        self.ln4 = LayerNorm(n_embd, bias=False)
        self.attn1 = RegionalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.attn2 = RegionalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size
        )
        self.cross_attn = RegionalCrossAttention(
            n_embd,
            n_heads,
            resid_pdrop,
            attn_pdrop,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, context):
        x = x + self.attn1(self.ln1(x), obj_att=False)  # temporal attention
        x = x + self.attn2(self.ln2(x), obj_att=True)   # object attention
        x = x + self.cross_attn(self.ln3(x), context)   # cross attention with language
        x = x + self.mlp(self.ln4(x))
        return x


class RegionalLanguageDecoder(nn.Module):
    def __init__(self, n_embd, n_heads, attn_pdrop, resid_pdrop, n_layers, n_bbox, block_size, out_dim=None, top_k: int =3, if_softmax=False):
        super().__init__()
        self.layers = nn.Sequential(*[AttentionBlock(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size)
                                      for _ in range(n_layers)])
        self.ln = LayerNorm(n_embd, bias=False)
        self.if_softmax = if_softmax

        if out_dim is not None:
            if if_softmax:
                self.top_k = top_k
                self.ft = nn.Linear(n_embd*top_k, out_dim)
            else:
                self.ft = nn.Linear(n_embd*n_bbox, out_dim)

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)
        out = self.ln(x)  # B, T, A, C

        if self.if_softmax:
            softmaxed_out = F.softmax(out, dim=-1)
            _, top_k_indices = torch.topk(softmaxed_out, self.top_k, dim=-2)
            out = torch.gather(out, -2, top_k_indices)

        out = out.view(out.size(0), out.size(1), -1)  # B, T, A*C

        if hasattr(self, 'ft'):
            out = self.ft(out)

        return out  # B, T, C

    def get_params(self):
        return self.parameters()
