import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
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
            A,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if obj_att:
            # B, T, A, C
            x = x.permute(0, 2, 1, 3)

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
                y.transpose(2, 3).transpose(1, 2).contiguous().view(B, A, T, C)
            )  # re-assemble all head outputs side by side
        else:
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
                y.transpose(2, 3).contiguous().view(B, A, T, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalSelfCrossAttention(nn.Module):
    def __init__(self, n_embd, cross_embed, n_heads, attn_pdrop, resid_pdrop, block_size):
        super().__init__()

        assert n_embd % n_heads == 0

        # Self-Attention Projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Cross-Attention Projections
        self.cross_key = nn.Linear(cross_embed, n_embd)
        self.cross_query = nn.Linear(n_embd, n_embd)
        self.cross_value = nn.Linear(cross_embed, n_embd)

        # Regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output Projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask for Self-Attention
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_head = n_heads

    def forward(self, x, cross_input=None):
        B, T, C = x.size()

        # calculate query, key, values for self-attention
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        if cross_input is not None:
            # calculate query, key, values for cross-attention
            T_C = cross_input.size(1)
            k_cross = self.cross_key(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)
            v_cross = self.cross_value(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)

            q_cross = self.cross_query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # cross-attention
            att_cross = (q_cross @ k_cross.transpose(-2, -1)) * (1.0 / math.sqrt(k_cross.size(-1)))
            att_cross = F.softmax(att_cross, dim=-1)
            att_cross = self.attn_drop(att_cross)
            y_cross = att_cross @ v_cross

            # combine self-attention and cross-attention
            y = y + y_cross  # or any other combination strategy

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            cross_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfCrossAttention(
            n_embd,
            cross_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, cond=None):
        x = x + self.attn(self.ln1(x), cross_input=cond)
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
            obj_att: bool
    ):
        super().__init__()

        self.obj_att = obj_att

        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = SelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), obj_att=self.obj_att)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        # self.blocks = nn.Sequential(
        #     *[EncoderBlock(
        #         embed_dim,
        #         n_heads,
        #         attn_pdrop,
        #         resid_pdrop,
        #         block_size,
        #     )
        #         for _ in range(n_layers)]
        # )

        self.blocks = nn.Sequential(
            EncoderBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
                obj_att=False
            ),
            EncoderBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
                obj_att=True
            ),
        )

        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            cross_embed: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DecoderBlock(
                embed_dim,
                cross_embed,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
            )
                for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None):
        for layer in self.blocks:
            x = layer(x, cond=cond)
        x = self.ln(x)
        return x


if __name__ == '__main__':

    # batch, agent, time, dimension
    dummy_input = torch.randn((1, 5, 1, 72))

    model = SelfAttention(72,
                          4,
                          0.1,
                          0.1,
                          block_size=5)

    out = model(dummy_input, obj_att=True)

    a = 0