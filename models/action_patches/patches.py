# Copyright 2024 The Genmo team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ActionTransformerBlock and ActionRotaryPosEmbed are derived from the LTX-Video
# transformer in diffusers (src/diffusers/models/transformers/transformer_ltx.py).
# Reached here by way of Genie-Envisioner (AgibotTech) at commit d54425c4, which
# did not carry the notice forward.
#
# Modified by the DexTacWAM Authors, 2026.

import math
from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class ActionRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Always compute rope in fp32
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)

        grid = grid / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs




@maybe_allow_in_graph
class ActionTransformerBlock(nn.Module):
    r"""
    Modified from Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        attn3_cross_attention_dim = 2048,
        num_latent_downsample_block = 0,
        dual_cross_attn: bool = False,
        tactile_gate_init: float = -3.0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(
            **(attention_args[0]),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        # Option A: dual cross-attn (cube_tactile_slowdown_ablation_1b64937f,
        # 2026-05-24 latest++++). When `dual_cross_attn` is False (legacy
        # shared-softmax path), a single `attn2` handles the concatenated
        # visual+tactile K/V. When True, `attn2_visual` consumes the visual
        # portion of `encoder_hidden_states` and `attn2_tactile` consumes the
        # tactile portion, with the tactile residual scaled by
        # `sigmoid(tactile_gate)` so the model can opt in to tactile as the
        # gate grows. `tactile_gate_init = -3.0` -> sigmoid ~ 0.047 at step 0,
        # which (a) gives the visual half a clean softmax denominator so it
        # can match Probe J's no-tactile convergence rate, while (b) leaving
        # enough gradient flow into the tactile branch that it can learn.
        # This is the Probe M intervention; see plan file
        # option-a-dual-crossattn_36ab43c4 for the M vs J vs K decision
        # matrix.
        self.dual_cross_attn = dual_cross_attn
        if self.dual_cross_attn:
            self.attn2_visual = attention_class(
                **(attention_args[1]),
            )
            self.attn2_tactile = attention_class(
                **(attention_args[1]),
            )
            self.tactile_gate = nn.Parameter(
                torch.full((), float(tactile_gate_init))
            )
        else:
            self.attn2 = attention_class(
                **(attention_args[1]),
            )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.num_latent_downsample_block = num_latent_downsample_block
        if self.num_latent_downsample_block > 0:
            self.latent_downsample_block = nn.ModuleList()
            for _i in range(self.num_latent_downsample_block):
                self.latent_downsample_block.append(
                    downsampling_block()
                )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_tactile: Optional[torch.Tensor] = None,
        attn3_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa


        if self.dual_cross_attn:
            # In dual mode `encoder_hidden_states` is the visual-only token
            # sequence (b, n_view_visual*L, c); the tactile token sequence
            # arrives separately as `encoder_hidden_states_tactile`
            # (b, n_view_tactile*L, c). The caller (model.forward via
            # `_split_action_kv`) is responsible for the split.
            # encoder_attention_mask currently unused along the dual path
            # because both branches see clean, padding-free token sequences
            # (n_view_visual / n_view_tactile are integer counts, not padded).
            #
            # When `encoder_hidden_states_tactile is None` the caller is
            # signaling "no tactile context for this forward" (validation /
            # `pipe.infer` path, or any other no-tactile inference). In
            # that case we run ONLY the visual cross-attn and skip the
            # tactile residual entirely. This matches Probe K's inference
            # behavior (no tactile injected at val) and avoids forcing
            # every inference codepath to manufacture a fake tactile K/V.
            attn_visual = self.attn2_visual(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
                attention_mask=None,
                n_view=1,
            )
            hidden_states = hidden_states + attn_visual

            if encoder_hidden_states_tactile is not None:
                attn_tactile = self.attn2_tactile(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states_tactile,
                    image_rotary_emb=None,
                    attention_mask=None,
                    n_view=1,
                )
                hidden_states = hidden_states + torch.sigmoid(self.tactile_gate) * attn_tactile
        else:
            attn_hidden_states = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
                attention_mask=encoder_attention_mask,
                n_view=1,
            )
            hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        

        return hidden_states



def add_action_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    action_in_channels: int = 14,
    action_out_channels: int = None,
    action_num_attention_heads: int = 16,
    action_attention_head_dim: int = 32,
    action_rope_dim: int = None,
    action_final_embeddings: bool = True,
    learnable_action_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class = None,
    attention_processor = None,
    dual_cross_attn: bool = False,
    tactile_gate_init: float = -3.0,
    **kwargs,
):

    if action_out_channels is None:
        action_out_channels = action_in_channels

    self.action_inner_dim = action_num_attention_heads * action_attention_head_dim

    self.learnable_action_state = learnable_action_state
    if self.learnable_action_state:
        self.action_state = nn.Parameter(torch.randn(1, 1, action_in_channels))

    self.action_proj_in = nn.Linear(action_in_channels, self.action_inner_dim)
    self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.action_inner_dim) / self.action_inner_dim**0.5)
    self.action_time_embed = AdaLayerNormSingle(self.action_inner_dim, use_additional_conditions=False)

    if action_rope_dim is None:
        action_rope_dim = self.action_inner_dim
    # set to a fixed value currently, should adjust according to the action length
    self.action_rope = ActionRotaryPosEmbed(
        dim=action_rope_dim,
        base_seq_length=57,
        theta=10000.0,
    )

    attention_args = []
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=None,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=inner_dim,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))

    self.action_blocks = nn.ModuleList(
        [
            ActionTransformerBlock(
                attention_class = attention_class,
                attention_args = attention_args,
                dim=self.action_inner_dim,
                num_attention_heads=action_num_attention_heads,
                attention_head_dim=action_attention_head_dim,
                cross_attention_dim=inner_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
                dual_cross_attn=dual_cross_attn,
                tactile_gate_init=tactile_gate_init,
            )
            for _ in range(num_layers)
        ]
    )

    self.action_proj_out = nn.Linear(self.action_inner_dim, action_out_channels) 
    self.action_final_embeddings = action_final_embeddings
    if not self.action_final_embeddings:
        self.action_proj_extra = nn.Linear(self.action_inner_dim, self.action_inner_dim)

    self.action_norm_out = nn.LayerNorm(self.action_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_action_states(
    self,
    action_states: torch.Tensor = None,
    action_timestep: torch.LongTensor = None,
):

    assert self.action_expert == True
    assert action_states is not None and action_timestep is not None

    batch_size = action_states.shape[0]

    action_seq_length = action_states.shape[1]

    if getattr(self, "learnable_action_state") and self.learnable_action_state:
        action_states = self.action_state.repeat(batch_size, action_seq_length, 1).to(dtype=action_states.dtype, device=action_states.device)

    action_rotary_emb = self.action_rope(action_states, action_seq_length)
    action_hidden_states = self.action_proj_in(action_states)

    action_temb, action_embedded_timestep = self.action_time_embed(
        action_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=action_hidden_states.dtype,
    )

    action_temb = action_temb.view(batch_size, -1, action_temb.size(-1))
    action_embedded_timestep = action_embedded_timestep.view(batch_size, -1, action_embedded_timestep.size(-1))
    
    return action_temb, action_embedded_timestep, action_rotary_emb, action_hidden_states
