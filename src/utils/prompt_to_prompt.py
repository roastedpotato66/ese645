"""
Prompt-to-Prompt attention control utilities.

Provides a controller that can capture cross/self attention maps from a reference
prompt run and re-inject them during editing to preserve structure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class PromptToPromptAttnProcessor:
    """
    Lightweight attention processor that mirrors `AttnProcessor` but applies
    controller-provided modifications to the attention probabilities.
    """

    def __init__(self, controller: "PromptToPromptController", place_in_unet: str):
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = self.controller.modify(attention_probs, attn.is_cross_attention, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class PromptToPromptController:
    """
    Stores attention maps from a reference prompt and replaces them during editing.
    """

    def __init__(
        self,
        self_replace_steps: float = 0.8,
        cross_replace_steps: float = 0.4,
    ):
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.mode: str = "record"
        self.total_steps: int = 0
        self.current_step: int = 0
        self.attention_store: Dict[Tuple[str, bool], List[Optional[torch.Tensor]]] = {}
        self.replace_token_mask: Optional[torch.Tensor] = None
        self.stats = {
            "record_calls": 0,
            "apply_calls": 0,
            "cross_replacements": 0,
            "self_replacements": 0,
        }

    def build_processors(self, unet) -> Dict[str, PromptToPromptAttnProcessor]:
        processors = {}
        for name in unet.attn_processors.keys():
            processors[name] = PromptToPromptAttnProcessor(self, name)
        return processors

    def set_total_steps(self, total_steps: int):
        self.total_steps = total_steps

    def set_mode(self, mode: str):
        if mode not in ("record", "apply"):
            raise ValueError("mode must be 'record' or 'apply'")
        self.mode = mode
        if mode == "record":
            self.attention_store = {}
            self.stats["record_calls"] = 0
            self.stats["apply_calls"] = 0
            self.stats["cross_replacements"] = 0
            self.stats["self_replacements"] = 0

    def set_step(self, step_index: int):
        self.current_step = step_index

    def set_replace_token_mask(self, mask: torch.Tensor):
        """
        mask: shape (1, 1, seq_len) where 1 means replace, 0 means keep target attention.
        """
        self.replace_token_mask = mask.clone()

    def modify(self, attention_probs: torch.Tensor, is_cross_attention: bool, place_in_unet: str) -> torch.Tensor:
        if self.mode == "record":
            self._save_attention(attention_probs, is_cross_attention, place_in_unet)
            self.stats["record_calls"] += 1
            return attention_probs

        self.stats["apply_calls"] += 1
        if not self._should_replace(is_cross_attention):
            return attention_probs

        key = (place_in_unet, is_cross_attention)
        store = self.attention_store.get(key)
        if store is None or self.current_step >= len(store):
            return attention_probs

        reference = store[self.current_step]
        if reference is None:
            return attention_probs

        reference = reference.to(attention_probs.device, attention_probs.dtype)
        if is_cross_attention and self.replace_token_mask is not None:
            mask = self.replace_token_mask[..., : reference.shape[-1]]
            mask = mask.to(attention_probs.device, attention_probs.dtype)
            attention_probs = reference * mask + attention_probs * (1.0 - mask)
        else:
            attention_probs = reference

        if is_cross_attention:
            self.stats["cross_replacements"] += 1
        else:
            self.stats["self_replacements"] += 1

        return attention_probs

    def _save_attention(self, attention_probs: torch.Tensor, is_cross_attention: bool, place_in_unet: str):
        if self.total_steps <= 0:
            return

        key = (place_in_unet, is_cross_attention)
        store = self.attention_store.setdefault(key, [None] * self.total_steps)
        store[self.current_step] = attention_probs.detach().float().cpu()

    def _should_replace(self, is_cross_attention: bool) -> bool:
        if self.total_steps <= 1:
            return False
        ratio = self.current_step / (self.total_steps - 1)
        threshold = self.cross_replace_steps if is_cross_attention else self.self_replace_steps
        return ratio <= threshold

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "total_steps": self.total_steps,
            "self_replace_steps": self.self_replace_steps,
            "cross_replace_steps": self.cross_replace_steps,
            "record_calls": self.stats["record_calls"],
            "apply_calls": self.stats["apply_calls"],
            "cross_replacements": self.stats["cross_replacements"],
            "self_replacements": self.stats["self_replacements"],
            "stored_layers": len(self.attention_store),
        }


