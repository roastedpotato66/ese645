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

        # MasaCtrl Hook: Allow modifying K/V (e.g. replacing with source)
        key, value = self.controller.modify_kv(key, value, attn.is_cross_attention, self.place_in_unet)

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
        store_self_attention: bool = True,
        attention_dtype: torch.dtype = torch.float32,
        step_stride: int = 1,
        layer_keywords: Optional[List[str]] = None,
        max_attention_size: int = 32 * 32,
        # MasaCtrl args
        use_masactrl: bool = False,
        masactrl_step_start: int = 0,
        masactrl_layer_keywords: Optional[List[str]] = None,
    ):
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.store_self_attention = store_self_attention
        self.attention_dtype = attention_dtype
        self.step_stride = max(1, step_stride)
        self.layer_keywords = layer_keywords
        self.max_attention_size = max_attention_size
        
        # MasaCtrl
        self.use_masactrl = use_masactrl
        self.masactrl_step_start = masactrl_step_start
        self.masactrl_layer_keywords = masactrl_layer_keywords
        self.kv_store: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

        self.mode: str = "record"
        self.total_steps: int = 0
        self.current_step: int = 0
        self.attention_store: Dict[Tuple[str, bool], Dict[int, torch.Tensor]] = {}
        self.replace_token_mask: Optional[torch.Tensor] = None
        self.stats = {
            "record_calls": 0,
            "apply_calls": 0,
            "cross_replacements": 0,
            "self_replacements": 0,
            "skipped_large": 0,
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
            self.kv_store = {}
            self.stats["record_calls"] = 0
            self.stats["apply_calls"] = 0
            self.stats["cross_replacements"] = 0
            self.stats["self_replacements"] = 0
            self.stats["skipped_large"] = 0

    def set_step(self, step_index: int):
        self.current_step = step_index

    def set_replace_token_mask(self, mask: torch.Tensor):
        """
        mask: shape (1, 1, seq_len) where 1 means replace, 0 means keep target attention.
        """
        self.replace_token_mask = mask.clone()

    def modify_kv(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        is_cross_attention: bool, 
        place_in_unet: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MasaCtrl implementation: Store (record) or Replace (apply) K and V matrices.
        """
        if not self.use_masactrl:
            return key, value
        
        # MasaCtrl typically targets Self-Attention
        if is_cross_attention:
            return key, value

        if not self._masactrl_layer_allowed(place_in_unet):
            return key, value

        if self.mode == "record":
            self._save_kv(key, value, place_in_unet)
            return key, value
        
        # Apply mode
        if self.current_step < self.masactrl_step_start:
            return key, value
            
        return self._replace_kv(key, value, place_in_unet)

    def _save_kv(self, key: torch.Tensor, value: torch.Tensor, place_in_unet: str):
        # Check size limit to prevent OOM (using same heuristic as attention store)
        seq_len = key.shape[2] # [Batch*Heads, Seq, Dim] after head_to_batch? 
        # Actually head_to_batch makes it [Batch*Heads, Seq, Dim]
        # Original shape [Batch, Heads, Seq, Dim]
        
        if seq_len > self.max_attention_size:
            return

        # Store only at relevant steps? If step_stride is used for attn, use it here too?
        # Usually MasaCtrl requires consistent replacement, so stride might break it.
        # But let's respect stride to save memory if configured, or assume stride=1 for MasaCtrl.
        
        # Store
        if place_in_unet not in self.kv_store:
            self.kv_store[place_in_unet] = {}
            
        # We store on CPU to save VRAM
        k_cpu = key.detach().cpu().to(self.attention_dtype)
        v_cpu = value.detach().cpu().to(self.attention_dtype)
        
        # Handle unconditional batching:
        # If batch size is 2*N (uncond+cond), we typically want the UNCOND or COND?
        # MasaCtrl paper usually uses the source image layout.
        # If source was generated with guidance, it has uncond+cond.
        # We usually take the COND part (second half) or UNCOND?
        # Actually, for structure preservation of the source image, we want the specific latents.
        # Standard P2P usually records the Uncond+Cond pass.
        # Let's store the whole thing for now.
        
        self.kv_store[place_in_unet][self.current_step] = (k_cpu, v_cpu)

    def _replace_kv(self, key: torch.Tensor, value: torch.Tensor, place_in_unet: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if place_in_unet not in self.kv_store:
            return key, value
        
        saved = self.kv_store[place_in_unet].get(self.current_step)
        if saved is None:
            return key, value
            
        k_src, v_src = saved
        k_src = k_src.to(key.device, dtype=key.dtype)
        v_src = v_src.to(value.device, dtype=value.dtype)
        
        # Dimensions: [Batch*Heads, Seq, Dim]
        # We want to replace.
        
        # If shapes mismatch (e.g. different batch size due to CFG settings), handle carefully.
        # Usually Inversion (Scale=1) has Batch=1 (or 2 if we use calc).
        # Inference (Scale=7.5) has Batch=2.
        
        # If Source was recorded with Scale=1 (Batch=1 or 2?)
        # In our DDIMInversion code, inversion uses `inversion_guidance_scale=1.0`.
        # But `predict_noise` implementation:
        # if scale==1.0 or uncond is None -> Batch=1 (only text).
        # So k_src has batch size corresponding to 1 input.
        
        # Target (Edit) uses Scale=7.5 -> Batch=2 (Uncond + Cond).
        
        # We want to replace K/V for BOTH Uncond and Cond passes in Target?
        # Or just Cond?
        # MasaCtrl usually replaces both to force layout in both components of CFG.
        
        if k_src.shape[0] != key.shape[0]:
            # Case: Source Batch=1 (Cond), Target Batch=2 (Uncond+Cond)
            # We need to duplicate src to match target
            if k_src.shape[0] * 2 == key.shape[0]:
                k_src = torch.cat([k_src, k_src], dim=0)
                v_src = torch.cat([v_src, v_src], dim=0)
            else:
                # Shape mismatch not easily resolvable
                return key, value
        
        return k_src, v_src

    def _masactrl_layer_allowed(self, place_in_unet: str) -> bool:
        if not self.masactrl_layer_keywords:
            # If none specified, MasaCtrl usually applies to specific decoder layers by default?
            # Or all? All is too much memory/constraint.
            # Default to "up" (decoder) layers if list is empty?
            # Or user must specify.
            # Let's allow "up" and "down" keywords.
            return False
        lowered = place_in_unet.lower()
        return any(keyword.lower() in lowered for keyword in self.masactrl_layer_keywords)

    def modify(self, attention_probs: torch.Tensor, is_cross_attention: bool, place_in_unet: str) -> torch.Tensor:
        if not self._layer_allowed(place_in_unet):
            return attention_probs

        if self.mode == "record":
            self._save_attention(attention_probs, is_cross_attention, place_in_unet)
            self.stats["record_calls"] += 1
            return attention_probs

        self.stats["apply_calls"] += 1
        if not self._should_replace(is_cross_attention):
            return attention_probs

        key = (place_in_unet, is_cross_attention)
        store = self.attention_store.get(key)
        if store is None:
            return attention_probs

        ref_step_idx = (self.current_step // self.step_stride) * self.step_stride
        reference = store.get(ref_step_idx)
        if reference is None:
            return attention_probs

        reference = reference.to(attention_probs.device, attention_probs.dtype)
        if attention_probs.shape[0] % 2 == 0:
            half = attention_probs.shape[0] // 2
            uncond_part = attention_probs[:half]
            cond_part = attention_probs[half:]

            if is_cross_attention and self.replace_token_mask is not None:
                mask = self.replace_token_mask[..., : reference.shape[-1]]
                mask = mask.to(cond_part.device, cond_part.dtype)
                cond_part = reference * mask + cond_part * (1.0 - mask)
            else:
                cond_part = reference

            attention_probs = torch.cat([uncond_part, cond_part], dim=0)
        else:
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
        seq_len = attention_probs.shape[-1]
        if seq_len > self.max_attention_size:
            self.stats["skipped_large"] += 1
            return

        if self.total_steps <= 0:
            return

        if not is_cross_attention and not self.store_self_attention:
            return

        if self.current_step % self.step_stride != 0:
            return

        key = (place_in_unet, is_cross_attention)
        if key not in self.attention_store:
            self.attention_store[key] = {}

        tensor = attention_probs.detach()
        if tensor.shape[0] % 2 == 0:
            half = tensor.shape[0] // 2
            tensor = tensor[half:]

        if tensor.dtype != self.attention_dtype:
            tensor = tensor.to(self.attention_dtype)
        tensor = tensor.cpu()

        self.attention_store[key][self.current_step] = tensor

    def _should_replace(self, is_cross_attention: bool) -> bool:
        if self.total_steps <= 1:
            return False
        ratio = self.current_step / (self.total_steps - 1)
        threshold = self.cross_replace_steps if is_cross_attention else self.self_replace_steps
        return ratio <= threshold

    def _layer_allowed(self, place_in_unet: str) -> bool:
        if not self.layer_keywords:
            return True
        lowered = place_in_unet.lower()
        return any(keyword.lower() in lowered for keyword in self.layer_keywords)

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "total_steps": self.total_steps,
            "self_replace_steps": self.self_replace_steps,
            "cross_replace_steps": self.cross_replace_steps,
             "store_self_attention": self.store_self_attention,
             "attention_dtype": str(self.attention_dtype),
             "step_stride": self.step_stride,
             "layer_keywords": self.layer_keywords,
            "record_calls": self.stats["record_calls"],
            "apply_calls": self.stats["apply_calls"],
            "cross_replacements": self.stats["cross_replacements"],
            "self_replacements": self.stats["self_replacements"],
            "skipped_large": self.stats["skipped_large"],
            "stored_layers": len(self.attention_store),
            "max_attention_size": self.max_attention_size,
        }


