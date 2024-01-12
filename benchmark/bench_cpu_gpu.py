import time
import math
import argparse
from typing import Optional, Tuple
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange
from timm.models.layers import trunc_normal_
from slidingchunk_2d import slidingchunk_2d, mask_invalid_locations, slidingchunk_2dautograd

class Long2DSCSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., w=7, d=1,
                 autoregressive=False, sharew=False, nglo=1, only_glo=False, exact=0, autograd=False, rpe=False, mode=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.Nglo = nglo
        self.only_glo = only_glo
        if self.only_glo:
            assert self.Nglo >= 1, "Nglo == 0 in the only global mode!"

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if nglo >= 1:
            if sharew:
                self.query_global = self.query
                self.kv_global = self.kv
                self.proj_global = self.proj
            else:
                self.query_global = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
                self.proj_global = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attention_window = w
        self.attention_dilation = d
        self.autoregressive = autoregressive

        assert self.attention_dilation == 1, "Dilation is not supported!"
        assert not self.autoregressive, "Autoregressive is not supported yet!"
        self.exact = exact
        # use autograd or handgrad
        self.longform2d_mm = slidingchunk_2dautograd if autograd else slidingchunk_2d

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * 2 * w - 1) * (2 * 2 * w - 1), num_heads))  # (4*w-1, 4*w-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(-w, 2*w)
            coords_w = torch.arange(-w, 2*w)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, 3w, 3w
            coords_unfold = rearrange(
                coords, 'c (m x) (n y) -> c m n (x y)', x=w, y=w
            )  # 2, 3, 3, 9w^2
            q_coords = coords_unfold[:, 1, 1, :] # 2, w^2
            relative_coords = torch.cat([
                # -1, -1
                q_coords[:, :, None] - coords_unfold[:, 0, 0, :][:, None, :],
                # -1, 0
                q_coords[:, :, None] - coords_unfold[:, 0, 1, :][:, None, :],
                # -1, 1
                q_coords[:, :, None] - coords_unfold[:, 0, 2, :][:, None, :],
                # 0,-1
                q_coords[:, :, None] - coords_unfold[:, 1, 0, :][:, None, :],
                # 0,0
                q_coords[:, :, None] - q_coords[:, None, :],
                # 0,1
                q_coords[:, :, None] - coords_unfold[:, 1, 2, :][:, None, :],
                # 1, -1
                q_coords[:, :, None] - coords_unfold[:, 2, 0, :][:, None, :],
                # 1, 0
                q_coords[:, :, None] - coords_unfold[:, 2, 1, :][:, None, :],
                # 1, 1
                q_coords[:, :, None] - coords_unfold[:, 2, 2, :][:, None, :],
            ], dim=-1)  # 2, w^2, 9w^2
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # w^2, 9w^2, 2
            relative_coords[:, :, 0] += 2 * w - 1  # shift to start from 0
            relative_coords[:, :, 1] += 2 * w - 1
            relative_coords[:, :, 0] *= 2 * 2 * w - 1
            relative_position_index = relative_coords.sum(-1)  # w^2, 9w^2
            self.register_buffer("relative_position_index", relative_position_index)

        # mode to control the sampling strategy of neighbor blocks
        # 0: all 8 blocks; -1: no neighbor block; >0: random sample one block
        self.mode = mode

    def forward(self, x, nx, ny):
        B, N, C = x.shape
        Nloc = nx * ny
        Nglo, H, M, W = self.Nglo, self.num_heads, self.head_dim, self.attention_window
        W2 = W ** 2
        assert Nglo + Nloc == N, "Global dimension does not match!"

        # get the mode of the longformer attention
        mode = self.mode
        kv_nums = 9 * W2
        if self.mode > 0:
            if self.training:
                mode = random.randrange(1, 9)  # 1 <= mode <= 8
                kv_nums = 2 * W2
            else:
                mode = 0  # full during evaluation
        elif mode == -1:
            kv_nums = W2

        # compute the local attention
        q = self.scale * self.query(x[:, Nglo:]).reshape(B, Nloc, H, M).transpose(1, 2).contiguous()
        kv = self.kv(x).reshape(B, N, 2, H, M).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        if self.only_glo:
            # local to global attn10: (B, self.num_heads, Nloc, Nglo)
            attn1 = torch.bmm(q.view(B*H, Nloc, M), k[:, :, :Nglo].reshape(B*H, Nglo, M).transpose(-2, -1)).view(B, H, Nloc, Nglo)
        else:
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b h (x y) c -> (b h) c x y', x=nx),
                (q, k[:, :, Nglo:], v[:, :, Nglo:]))
            # pad 0's to make sure that nx % W == 0, ny % W == 0
            (padx, pady) = map(lambda t: (W - t % W) % W, (nx, ny))
            (mx, my) = map(lambda t: (t[0] + t[1]) // W,
                           ((nx, padx), (ny, pady)))
            if padx > 0 or pady > 0:
                (q_img, k_img, v_img) = map(
                    lambda t: F.pad(t, (0, pady, 0, padx)), (q_img, k_img, v_img)
                )
            # unfold the padded tensor
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b c (m x) (n y) -> b c m n (x y)', x=W, y=W),
                (q_img, k_img, v_img)
            )

            # local to global attn10: (B*H, mx, my, w^2, Nglo)
            attn10 = einsum('b c m n l, b t c -> b m n l t', q_img,
                       k[:, :, :Nglo].reshape(B*H, Nglo, M))
            # local to local attn11： (B*H, mx, my, W**2, 9*W**2), mode = 0
            # attn11： (B*H, mx, my, W**2, W**2), mode = -1
            # attn11： (B*H, mx, my, W**2, 2*W**2), mode > 0
            attn11 = self.longform2d_mm(q_img, k_img, False, mode)

            if self.rpe:
                if Nglo >= 1:
                    # local to global bias
                    attn10 = attn10 + self.g2l_relative_position_bias[1].unsqueeze(0).expand(B, -1, -1).reshape(B*H, Nglo)[:, None, None, None, :]
                # local to local bias
                if mode == -1:
                    relative_position_index = self.relative_position_index[:, 4 * W2:5 * W2].contiguous()
                elif mode == 0:
                    relative_position_index = self.relative_position_index
                else:  # mode > 0
                    chunk_id = mode if mode > 4 else mode - 1
                    relative_position_index = torch.cat([
                        self.relative_position_index[:, 4 * W2:5 * W2],
                        self.relative_position_index[:, chunk_id * W2:(chunk_id+1) * W2],
                    ], dim=-1)
                local_relative_position_bias = self.local_relative_position_bias_table[
                    relative_position_index.view(-1)].view(1, W2, kv_nums, -1)  # w^2, kv_nums,H
                local_relative_position_bias = local_relative_position_bias.permute(
                    0, 3, 1, 2).expand(B, -1, -1, -1).contiguous().view(B*H, W2, kv_nums)  # B*H, w^2, kv_nums
                attn11 = attn11 + local_relative_position_bias[:, None, None, :, :]

            num_invalid = mask_invalid_locations(
                attn11, mx, my, padx, pady, W, exact=self.exact, mode=mode
            )
            attn1 = torch.cat((attn10, attn11), dim=-1)

        attn1 = (attn1 - torch.max(attn1, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # update x1: (B, self.num_heads, Nloc, self.head_dim)
        if self.only_glo:
            x1 = torch.bmm(
                attn1.view(B * H, Nloc, Nglo), v[:, :, :Nglo].reshape(B * H, Nglo, M)
            ).view(B, H, Nloc, M)
        else:
            attnl2g = attn1[:, :, :, :, :Nglo]
            x1 = self.longform2d_mm(attn1[:, :, :, :, Nglo:Nglo+kv_nums], v_img, True, mode)
            if Nglo >= 1:
                x1 = x1 + einsum(
                    'b m n l t, b t c -> b c m n l', attnl2g,
                    v[:, :, :Nglo].reshape(B * H, Nglo, M)
                )
            x1 = rearrange(x1, 'b c m n (x y) -> b (m x) (n y) c', x=W)
            x1 = x1[:, :nx, :ny].reshape(B, H, Nloc, M)
        x1 = x1.transpose(1, 2).reshape(B, Nloc, C)
        x1 = self.proj(x1)

        if Nglo == 0:
            return self.proj_drop(x1)

        # compute the glocal attention; same with vanilla multi-head attention
        q_global = self.scale * self.query_global(x[:, :Nglo]).reshape(B, Nglo, H, M).transpose(1, 2)
        kv_global = self.kv_global(x).reshape(B, N, 2, H, M).permute(2, 0, 3, 1, 4)
        k_global, v_global = kv_global[0], kv_global[1]  # make torchscript happy (cannot use tensor as tuple)
        # attention matrix
        attn0 = torch.bmm(q_global.reshape(B*H, Nglo, M), k_global.reshape(B*H, N, M).transpose(-2, -1))
        if self.rpe:
            # relative position embedding of global tokens
            global_relative_position_bias = torch.cat([
                self.g2g_relative_position_bias,
                self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, Nloc)
            ], dim=-1)  # nH, nglo, N
            attn0 = attn0 + global_relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B*H, Nglo, N)

        attn0 = (attn0 - torch.max(attn0, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn0 = self.attn_drop(attn0)
        # context vector
        x0 = torch.bmm(attn0, v_global.reshape(B*H, N, M)).view(B, H, Nglo, M).transpose(1, 2).reshape(B, Nglo, C)
        x0 = self.proj_global(x0)

        return self.proj_drop(torch.cat((x0, x1), dim=1))

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        S = T
        Nglo, H, M, W = module.Nglo, module.num_heads, module.head_dim, module.attention_window
        macs = 0
        n_params = 0

        # Sliding window scaled-dot-product macs
        if module.only_glo:
            # local to global
            # [B x T x (C-Nglo)] x [B x C x Nglo] --> [B x T x Nglo]
            num_macs_kq = (C - Nglo) * Nglo * C
        else:
            # local to local
            # [B x T x (C-Nglo)] x [B x C x (S-Nglo)] --> [B x (C-Nglo) x (9 * W**2)]
            num_macs_kq = (C-Nglo) * (9 * W**2) * C
            # local to global
            # [B x T x (C-Nglo)] x [B x C x Nglo] --> [B x T x Nglo]
            num_macs_kq += (C-Nglo) * Nglo * C
        # global to all
        # [B x T x Nglo] x [B x C x S] --> [B x Nglo x S]
        num_macs_kq += Nglo * S * C
        # same computational cost for attn * v -> context
        num_macs_v = num_macs_kq

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        # by default, we share weights for local and global tokens
        q_params = sum([p.numel() for p in module.query.parameters()])
        kv_params = sum([p.numel() for p in module.kv.parameters()])
        n_params += q_params + kv_params
        # multiply by Seq length
        macs += (q_params + kv_params) * T
        # print('macs qkv', qkv_params * T / 1e8)

        # by default, we share weights for local and global tokens
        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class LongformerSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_window, layer_id):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_heads = num_attention_heads
        self.head_dim = int(hidden_size / num_attention_heads)
        self.embed_dim = hidden_size

        self.query = nn.Linear(hidden_size, self.embed_dim)
        self.key = nn.Linear(hidden_size, self.embed_dim)
        self.value = nn.Linear(hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(hidden_size, self.embed_dim)
        self.key_global = nn.Linear(hidden_size, self.embed_dim)
        self.value_global = nn.Linear(hidden_size, self.embed_dim)

        self.dropout = 0.1

        self.layer_id = layer_id
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.
        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:
            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        # hidden_states = hidden_states.unsqueeze(dim=0)
        hidden_states = hidden_states.transpose(0, 1)


        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        # remove_from_windowed_attention_mask = (attention_mask != 0)

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example:
        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```
                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        if not onnx_export:
            # non-overlapping chunks of size = 2w
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

        # When exporting to ONNX, use this separate logic
        # have to use slow implementation since as_strided, unfold and 2d-tensor indexing aren't supported (yet) in ONNX export

        # TODO replace this with
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # once `unfold` is supported
        # the case hidden_states.size(1) == window_overlap * 2 can also simply return hidden_states.unsqueeze(1), but that's control flow

        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap, False)
        key = self._chunk(key, window_overlap, False)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))

        assert list(global_attn_scores.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], (
            "global_attn_scores have the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
            f" {global_attn_scores.size()}."
        )

        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            global_attn_probs_float = layer_head_mask.view(1, -1, 1, 1) * global_attn_probs_float.view(
                batch_size, self.num_heads, max_num_global_attn_indices, seq_len
            )
            global_attn_probs_float = global_attn_probs_float.view(
                batch_size * self.num_heads, max_num_global_attn_indices, seq_len
            )

        global_attn_probs = nn.functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], (
            "global_attn_output tensor has the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
            f" {global_attn_output.size()}."
        )

        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output, global_attn_probs

class BertSelfAttention(nn.Module):
	def __init__(self, hidden_size, num_attention_heads, including_proj, attention_probs_dropout_prob=0.1):
		super(BertSelfAttention, self).__init__()
		if hidden_size % num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))
		self.num_attention_heads = num_attention_heads
		self.attention_head_size = int(hidden_size / num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		
		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.including_proj = including_proj

		self.dropout = nn.Dropout(attention_probs_dropout_prob)
		
	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = torch.reshape(x, new_x_shape)
		return x.permute(0, 2, 1, 3)
	
	def transpose_key_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = torch.reshape(x, new_x_shape)
		return x.permute(0, 2, 3, 1)
	
	def forward(self, hidden_states, attention_mask):
		# assume attention_mask: [batch_size, um_attention_heads, seq_len, seq_len]
		
		# hidden_states: [batch_size, seq_len, config.hidden_size]
		# mixed_*_layer: [batch_size, seq_len, num_attention_heads * attention_head_size = config.hidden_size]
		if self.including_proj:
			mixed_query_layer = self.query(hidden_states)
			mixed_key_layer = self.key(hidden_states)
			mixed_value_layer = self.value(hidden_states)
            
            # {q,v}_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]
            # key_layer:   [batch_size, num_attention_heads, attention_head_size, seq_len]
			query_layer = self.transpose_for_scores(mixed_query_layer)
			key_layer = self.transpose_key_for_scores(mixed_key_layer)
			value_layer = self.transpose_for_scores(mixed_value_layer)
		else:
			query_layer = hidden_states[0]
			key_layer = hidden_states[1]
			value_layer = hidden_states[2]
			
		# Take the dot product between "query" and "key" to get the raw attention scores.
		# attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
		attention_scores = torch.matmul(query_layer, key_layer)
			
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
			
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask
		
		# Normalize the attention scores to probabilities.
		attention_probs = F.softmax(attention_scores, dim=-1)
			
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		
		# context_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]
		context_layer = torch.matmul(attention_probs, value_layer)
		if self.including_proj:
		    # context_layer: [batch_size, seq_len, num_attention_heads, attention_head_size]
			context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            # context_layer: [batch_size, seq_len, num_attention_heads * attention_head_size = config.hidden_size]
			new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

			context_layer = torch.reshape(context_layer, new_context_layer_shape)
			context_layer = self.dense(context_layer)
		return context_layer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GPT2SelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, attn_pdrop, resid_pdrop, scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        # if only "normal" attention layer implements causal mask
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = torch.matmul(w, v)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value, attention_mask)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a


class BartSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output

def build_ViL_model_and_input(batch_size=1, seq_len=56, use_large=False, cuda=True, fp16=False, including_proj=True):
    if including_proj:
        if not use_large:
            hidden_size, num_heads = 384, 12
            attention_window = 15
        else:
            hidden_size, num_heads = 1024, 16

        model = Long2DSCSelfAttention(dim=hidden_size, w=attention_window).eval()
        hidden_state = torch.randn(batch_size, seq_len * seq_len + 1, hidden_size)

        if fp16:
            model = model.half()
            hidden_state = hidden_state.half()

        #attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
        
        if cuda:
            model = model.cuda()
            hidden_state = hidden_state.cuda()
    else:
        if not use_large:
            hidden_size, num_heads = 768, 12
        else:
            hidden_size, num_heads = 1024, 16

        model = BertSelfAttention(hidden_size, num_heads, including_proj).eval()
        q = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        v = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        k = torch.randn(batch_size, num_heads, int(hidden_size / num_heads), seq_len)

        if fp16:
            model = model.half()
            q = q.half()
            k = k.half()
            v = v.half()

        attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
        
        if cuda:
            model = model.cuda()
            q = q.cuda()
            k = k.cuda()
            v = v.cuda()
            attn_mask = attn_mask.cuda()
        
        hidden_state = [q,k,v]
    
    return model, (hidden_state, 28, 28)


def build_longformer_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False, including_proj=True):
    if including_proj:
        if not use_large:
            hidden_size, num_heads = 768, 12
            attention_window = 512
        else:
            hidden_size, num_heads = 1024, 16

        model = LongformerSelfAttention(hidden_size, num_heads, attention_window, 1).eval()
        hidden_state = torch.randn((batch_size, seq_len, hidden_size))

        if fp16:
            model = model.half()
            hidden_state = hidden_state.half()

        attn_mask = torch.randn(batch_size, seq_len).long()
        is_index_masked = torch.randn(batch_size, seq_len).bool()
        
        if cuda:
            model = model.cuda()
            hidden_state = hidden_state.cuda()
            attn_mask = attn_mask.cuda()
            is_index_masked = is_index_masked.cuda()
    else:
        if not use_large:
            hidden_size, num_heads = 768, 12
        else:
            hidden_size, num_heads = 1024, 16

        model = BertSelfAttention(hidden_size, num_heads, including_proj).eval()
        q = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        v = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        k = torch.randn(batch_size, num_heads, int(hidden_size / num_heads), seq_len)

        if fp16:
            model = model.half()
            q = q.half()
            k = k.half()
            v = v.half()

        attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
        
        if cuda:
            model = model.cuda()
            q = q.cuda()
            k = k.cuda()
            v = v.cuda()
            attn_mask = attn_mask.cuda()
        
        hidden_state = [q,k,v]
    
    return model, (hidden_state, attn_mask, None, is_index_masked)

def build_bert_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False, including_proj=True):
    if including_proj:
        if not use_large:
            hidden_size, num_heads = 768, 12
        else:
            hidden_size, num_heads = 1024, 16

        model = BertSelfAttention(hidden_size, num_heads, including_proj).eval()
        hidden_state = torch.randn(batch_size, seq_len, hidden_size)

        if fp16:
            model = model.half()
            hidden_state = hidden_state.half()

        attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
        
        if cuda:
            model = model.cuda()
            hidden_state = hidden_state.cuda()
            attn_mask = attn_mask.cuda()
    else:
        if not use_large:
            hidden_size, num_heads = 768, 12
        else:
            hidden_size, num_heads = 1024, 16

        model = BertSelfAttention(hidden_size, num_heads, including_proj).eval()
        q = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        v = torch.randn(batch_size, num_heads, seq_len, int(hidden_size / num_heads))
        k = torch.randn(batch_size, num_heads, int(hidden_size / num_heads), seq_len)

        if fp16:
            model = model.half()
            q = q.half()
            k = k.half()
            v = v.half()

        attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
        
        if cuda:
            model = model.cuda()
            q = q.cuda()
            k = k.cuda()
            v = v.cuda()
            attn_mask = attn_mask.cuda()
        
        hidden_state = [q,k,v]
    
    return model, (hidden_state, attn_mask)


def build_gpt2_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    attn_pdrop, resid_pdrop, scale = 0.1, 0.1, True
    if not use_large:
        n_embed, n_ctx, n_head = 768, 1024, 12
    else:
        n_embed, n_ctx, n_head = 1024, 1024, 16

    model = GPT2SelfAttention(n_embed, n_ctx, n_head, attn_pdrop, resid_pdrop, scale)
    hidden_state = torch.randn(batch_size, seq_len, n_embed)

    if fp16:
        model = model.half()
        hidden_state = hidden_state.half()

    attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()

    if cuda:
        model = model.cuda()
        hidden_state = hidden_state.cuda()
        attn_mask = attn_mask.cuda()

    return model, (hidden_state, attn_mask)


def build_bart_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    if not use_large:
        # d_model, encoder_attention_heads, attention_dropout
        embed_dim, num_heads, dropout = 768, 12, 0.1
    else:
        embed_dim, num_heads, dropout = 1024, 16, 0.1

    model = BartSelfAttention(embed_dim, num_heads, dropout)
    hidden_state = torch.randn(batch_size, seq_len, embed_dim)

    if fp16:
        model = model.half()
        hidden_state = hidden_state.half()

    if cuda:
        model = model.cuda()
        hidden_state = hidden_state.cuda()

    return model, (hidden_state,)


def bench_dense_attn_cpu(run_func, number=10, repeats=10):
    run_func()
    bench_res = []
    
    for i in range(repeats):
        time_record = []
        
        for j in range(number):
            tic = time.time()
            run_func()
            toc = time.time()
            time_record.append(1000 * (toc - tic))

        bench_res.append(np.mean(time_record))
    
    return bench_res


def bench_dense_attn_gpu(run_func, number=1000, repeats=10):
    run_func()
    bench_res = []

    for i in range(repeats):
        time_record = []
        
        for j in range(number):
            torch.cuda.synchronize()
            
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            
            tic.record()
            
            run_func()

            toc.record()
            torch.cuda.synchronize()

            elapsed = tic.elapsed_time(toc)
            time_record.append(elapsed)
        
        avg_time = np.mean(time_record)
        bench_res.append(avg_time)

    return bench_res


def run_dense_attn(dense_attn, inputs):
    with torch.no_grad():
        output = dense_attn(*inputs)

def run_ViL_benchmark(batch_size=1, seq_len=4096, use_large=False, cuda=True, fp16=False, including_proj=True):
    dense_attn, inputs = build_ViL_model_and_input(batch_size=batch_size, seq_len=seq_len, use_large=use_large, cuda=cuda, fp16=fp16, including_proj=including_proj)
    run_func = partial(run_dense_attn, dense_attn=dense_attn, inputs=inputs)
    if cuda:
        bench_res = bench_dense_attn_gpu(run_func)
    else:
        bench_res = bench_dense_attn_cpu(run_func)
    print(f"Benchmark result ({'ViL'}, {'GPU' if cuda else 'CPU'}, {'TC' if fp16 else 'NTC'}, {seq_len})")
    print(bench_res)
    print(f"mean: {np.mean(bench_res)}, std: {np.std(bench_res)}")
    return np.mean(bench_res)

def run_longformer_benchmark(batch_size=1, seq_len=4096, use_large=False, cuda=True, fp16=False, including_proj=True):
    dense_attn, inputs = build_longformer_model_and_input(batch_size=batch_size, seq_len=seq_len, use_large=use_large, cuda=cuda, fp16=fp16, including_proj=including_proj)
    run_func = partial(run_dense_attn, dense_attn=dense_attn, inputs=inputs)
    if cuda:
        bench_res = bench_dense_attn_gpu(run_func)
    else:
        bench_res = bench_dense_attn_cpu(run_func)
    print(f"Benchmark result ({'Longformer'}, {'GPU' if cuda else 'CPU'}, {'TC' if fp16 else 'NTC'}, {seq_len})")
    print(bench_res)
    print(f"mean: {np.mean(bench_res)}, std: {np.std(bench_res)}")
    return np.mean(bench_res)

def main():
    # ViL-stage-1 GPU
    run_ViL_benchmark(seq_len=56)
    # ViL-stage-1 CPU
    run_ViL_benchmark(seq_len=56, cuda=False)

    # ViL-stage-2 GPU
    run_ViL_benchmark(seq_len=28)
    # ViL-stage-2 CPU
    run_ViL_benchmark(seq_len=28, cuda=False)

    # Longformer GPU
    run_longformer_benchmark()
    # Longformer CPU
    run_longformer_benchmark(cuda=False)


if __name__ == '__main__':
    main()
