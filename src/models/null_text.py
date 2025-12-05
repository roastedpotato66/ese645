"""
Null-Text Inversion implementation.
Simplified version that relies on the updated PromptToPromptController for MasaCtrl masking.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import math

from src.models.ddim import (
    DEFAULT_MODEL_ID,
    DDIMConfig,
    DDIMInversion,
    StandardImageEditingSetup,
)
from src.models.registry import register_model
from src.utils.ddim_utils import (
    decode_latent,
    ddim_step_forward,
    encode_prompt,
    load_image,
    predict_noise,
    set_seed,
    tensor_to_pil,
)
from src.utils.prompt_to_prompt import PromptToPromptController


# ==========================================
# Helper: RLE Mask Decoder
# ==========================================
def decode_rle_mask(rle_data: List[int], height: int = 512, width: int = 512) -> torch.Tensor:
    """
    Decodes COCO-style RLE mask data into a binary torch tensor.
    """
    total_pixels = height * width
    mask = torch.zeros(total_pixels, dtype=torch.float32)
    
    current_pos = 0
    val = 0 
    
    for count in rle_data:
        if current_pos + count > total_pixels:
            count = total_pixels - current_pos
            
        if val == 1:
            mask[current_pos : current_pos + count] = 1.0
            
        current_pos += count
        val = 1 - val
        
    mask = mask.reshape(height, width)
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class NullTextConfig(DDIMConfig):
    """
    Configuration for Null-Text Inversion.
    """
    null_inner_steps: int = 10
    null_lr: float = 0.01

    cross_replace_steps: float = 0.0
    self_replace_steps: float = 0.0

    # Latent Blending
    use_latent_blending: bool = False
    latent_blend_steps: int = 0

    # MasaCtrl
    use_masactrl: bool = False
    masactrl_step_start: int = 15
    masactrl_layer_keywords: List[str] = field(default_factory=lambda: [
        "output_blocks.1.attentions.1",
        "output_blocks.2.attentions.1",
        "output_blocks.3.attentions.1",
    ])


class NullTextInversion:
    def __init__(self, setup: StandardImageEditingSetup, config: NullTextConfig):
        self.setup = setup
        self.config = config
        self.base_inversion = DDIMInversion(setup, config)

    def invert(self, image: Any, source_prompt: str) -> Dict[str, Any]:
        inversion_result = self.base_inversion.invert(image, source_prompt)
        
        optimized_uncond_embeddings = self.optimize_null_text(
            inversion_result["latents"],
            inversion_result["text_embeddings"],
            inversion_result["uncond_embeddings"]
        )
        inversion_result["optimized_uncond_embeddings"] = optimized_uncond_embeddings
        return inversion_result

    def optimize_null_text(self, latents, text_embeddings, initial_uncond_embeddings):
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        optimized_embeddings = []
        timesteps_list = list(scheduler.timesteps)
        
        current_null_embedding = initial_uncond_embeddings.clone().detach()
        current_null_embedding.requires_grad = True
        current_latent = latents[-1]
        
        pbar = tqdm(total=len(timesteps_list), desc="Null-Text Optimization", leave=False, position=1, file=sys.stderr)
        
        for i, timestep in enumerate(timesteps_list):
            target_latent = latents[len(latents) - 2 - i]
            current_lr = self.config.null_lr * (1.0 - i / len(timesteps_list))
            optimizer = Adam([current_null_embedding], lr=current_lr)
            
            for _ in range(self.config.null_inner_steps):
                optimizer.zero_grad()
                latent_input = torch.cat([current_latent] * 2)
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                combined_embeddings = torch.cat([current_null_embedding, text_embeddings])
                
                noise_pred = self.setup.unet(latent_input, timestep, encoder_hidden_states=combined_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                prev_latent_pred = ddim_step_forward(scheduler, noise_pred_cfg, timestep, current_latent)
                loss = F.mse_loss(prev_latent_pred, target_latent)
                loss.backward()
                optimizer.step()
            
            optimized_embeddings.append(current_null_embedding.detach().clone())
            
            # Next Step
            with torch.no_grad():
                latent_input = torch.cat([current_latent] * 2)
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                combined_embeddings = torch.cat([current_null_embedding, text_embeddings])
                noise_pred = self.setup.unet(latent_input, timestep, encoder_hidden_states=combined_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                current_latent = ddim_step_forward(scheduler, noise_pred_cfg, timestep, current_latent)
            
            current_null_embedding = current_null_embedding.detach()
            current_null_embedding.requires_grad = True
            pbar.update(1)
        
        pbar.close()
        return optimized_embeddings

    @torch.no_grad()
    def edit(self, inversion_result, target_prompt, attention_controller=None):
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        if attention_controller:
            attention_controller.set_total_steps(len(scheduler.timesteps))

        zt = inversion_result["latents"][-1]
        source_latents = inversion_result["latents"]
        target_embeddings = encode_prompt(self.setup.text_encoder, self.setup.tokenizer, target_prompt, self.setup.device)
        optimized_uncond = inversion_result["optimized_uncond_embeddings"]

        for idx, timestep in enumerate(scheduler.timesteps):
            if attention_controller:
                attention_controller.set_step(idx)
            
            uncond_embeddings = optimized_uncond[idx] if idx < len(optimized_uncond) else inversion_result["uncond_embeddings"]

            # Latent Blending
            if self.config.use_latent_blending and idx < self.config.latent_blend_steps:
                target_idx = -(idx + 2)
                if abs(target_idx) <= len(source_latents):
                    zt = source_latents[target_idx]
                    continue

            noise_pred = predict_noise(self.setup.unet, zt, timestep, target_embeddings, self.config.guidance_scale, uncond_embeddings)
            zt = ddim_step_forward(scheduler, noise_pred, timestep, zt)

        return decode_latent(self.setup.vae, zt, self.setup.device)


@register_model("null_text")
class NullTextEditor:
    def __init__(self, device="auto", model_id=DEFAULT_MODEL_ID, num_inference_steps=None, 
                 num_inversion_steps=None, guidance_scale=None, seed=None, precision=None, 
                 null_inner_steps=10, null_lr=1e-2, config_overrides=None):
        
        resolved_device = self._resolve_device(device)
        dtype = self._resolve_dtype(resolved_device, precision)
        
        self.config = NullTextConfig()
        if num_inference_steps: self.config.num_inference_steps = num_inference_steps
        if num_inversion_steps: self.config.num_inversion_steps = num_inversion_steps
        if guidance_scale: self.config.guidance_scale = guidance_scale
        if seed: self.config.seed = seed
        self.config.null_inner_steps = null_inner_steps
        self.config.null_lr = null_lr
        self.config.dtype = dtype
        self.config.update(config_overrides)
        
        set_seed(self.config.seed)
        self.setup = StandardImageEditingSetup(model_id=model_id, device=resolved_device, dtype=dtype)
        self.inversion = NullTextInversion(self.setup, self.config)

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            if torch.cuda.is_available(): return torch.device("cuda")
            if torch.backends.mps.is_available(): return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(device, precision):
        if precision == "fp32" or device.type == "cpu": return torch.float32
        return torch.float16
    
    @staticmethod
    def _resolve_attention_dtype(name):
        return torch.float32 if "32" in name else torch.float16

    def edit_image(
        self,
        image_path: str,
        prompt_src: str,
        prompt_tar: str,
        blend_word: Optional[str] = None,
        provided_mask: Optional[List[int]] = None,  # RLE data from JSON
        output_path: Optional[str] = None,
        return_intermediate: bool = False,
        verbose: bool = False,
        # Overrides
        use_masactrl: Optional[bool] = None,
        masactrl_step_start: Optional[int] = None,
        use_latent_blending: Optional[bool] = None,
        latent_blend_steps: Optional[int] = None,
        **kwargs,
    ):
        # 1. Update Config Overrides
        original_config = {}
        if use_masactrl is not None: 
            original_config['use_masactrl'] = self.config.use_masactrl
            self.config.use_masactrl = use_masactrl
        if masactrl_step_start is not None:
            original_config['masactrl_step_start'] = self.config.masactrl_step_start
            self.config.masactrl_step_start = masactrl_step_start
        if use_latent_blending is not None:
            original_config['use_latent_blending'] = self.config.use_latent_blending
            self.config.use_latent_blending = use_latent_blending
        if latent_blend_steps is not None:
            original_config['latent_blend_steps'] = self.config.latent_blend_steps
            self.config.latent_blend_steps = latent_blend_steps

        try:
            image = load_image(image_path, self.config.image_size)
            if verbose: print("Starting Inversion...")
            inversion_result = self.inversion.invert(image, prompt_src)

            # === MASK LOGIC ===
            masactrl_mask = None
            if self.config.use_masactrl and provided_mask is not None:
                if verbose: print("Decoding provided RLE mask for MasaCtrl...")
                # Decode: 1=Object, 0=Background
                object_mask = decode_rle_mask(provided_mask, self.config.image_size, self.config.image_size)
                object_mask = object_mask.to(self.setup.device)
                
                # Invert for MasaCtrl: We need 1=Background (Keep), 0=Object (Change)
                masactrl_mask = 1.0 - object_mask
            
            # Fallback to auto-computation if no RLE provided but blend_word exists
            elif self.config.use_masactrl and blend_word:
                if verbose: print(f"Computing spatial mask for '{blend_word}'...")
                masactrl_mask = self._compute_spatial_mask(
                    inversion_result["latents"][0],
                    prompt_src,
                    blend_word
                )

            # Controller Setup
            controller = self._build_controller(prompt_tar, blend_word)
            
            if controller:
                # Inject mask into controller (PromptToPromptController now supports this natively)
                if hasattr(controller, 'set_masactrl_mask') and masactrl_mask is not None:
                    controller.set_masactrl_mask(masactrl_mask)
                
                with self._use_attention_controller(controller):
                    # Phase 1: Record (Disable blending)
                    controller.set_mode("record")
                    blend_backup = self.config.use_latent_blending
                    self.config.use_latent_blending = False
                    
                    self.inversion.edit(inversion_result, prompt_src, attention_controller=controller)
                    
                    # Phase 2: Apply (Restore blending)
                    self.config.use_latent_blending = blend_backup
                    controller.set_mode("apply")
                    
                    edited_tensor = self.inversion.edit(inversion_result, prompt_tar, attention_controller=controller)
            else:
                edited_tensor = self.inversion.edit(inversion_result, prompt_tar)

            # Save/Return
            reconstructed_tensor = decode_latent(self.setup.vae, inversion_result["latents"][0], self.setup.device)
            source_pil = image
            reconstructed_pil = tensor_to_pil(reconstructed_tensor)
            edited_pil = tensor_to_pil(edited_tensor)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                edited_pil.save(output_path)

            if return_intermediate:
                return {"source": source_pil, "reconstructed": reconstructed_pil, "edited": edited_pil}
            return edited_pil

        finally:
            for k, v in original_config.items(): setattr(self.config, k, v)

    def _build_controller(self, prompt, blend_word):
        if not self.config.use_prompt_to_prompt and not self.config.use_masactrl: return None
        
        controller = PromptToPromptController(
            self_replace_steps=self.config.self_replace_steps,
            cross_replace_steps=self.config.cross_replace_steps,
            attention_dtype=self._resolve_attention_dtype(self.config.p2p_attention_dtype),
            use_masactrl=self.config.use_masactrl,
            masactrl_step_start=self.config.masactrl_step_start,
            masactrl_layer_keywords=self.config.masactrl_layer_keywords,
        )
        return controller

    @contextmanager
    def _use_attention_controller(self, controller):
        """
        Simplified context manager since PromptToPromptController now handles 
        masked processors natively in build_processors.
        """
        if controller is None:
            yield
            return
        
        original_processors = dict(self.setup.unet.attn_processors)
        # The controller.build_processors() will now return the updated 
        # PromptToPromptAttnProcessor which includes the Masked MasaCtrl logic.
        processors = controller.build_processors(self.setup.unet)
        
        self.setup.unet.set_attn_processor(processors)
        try:
            yield
        finally:
            self.setup.unet.set_attn_processor(original_processors)

    def _compute_spatial_mask(self, latent: torch.Tensor, prompt: str, word: str) -> Optional[torch.Tensor]:
        """
        Fallback: Cross-Attention based mask computation.
        """
        # 1. Encode prompt
        text_input = self.setup.tokenizer(
            prompt, padding="max_length", max_length=self.setup.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings = self.setup.text_encoder(text_input.input_ids.to(self.setup.device))[0]
        
        # 2. Find word indices (Updated to handle "word word" case)
        word_indices = []
        tokens = self.setup.tokenizer.convert_ids_to_tokens(text_input.input_ids[0])
        target_words = [w.strip().lower() for w in word.replace(",", " ").split() if w.strip()]
        
        for idx, token in enumerate(tokens):
            clean_token = token.replace("Ġ", "").replace("</w>", "").lower()
            if clean_token in target_words:
                word_indices.append(idx)
        
        if not word_indices:
            print(f"Warning: Words {target_words} not found in prompt. MasaCtrl will apply globally.")
            return None

        # 3. Capture Cross-Attention maps
        attn_maps = []
        def hook_fn(module, input, output):
            if output.shape[-1] == 77: 
                attn_maps.append(output.detach().cpu())

        handles = []
        for name, module in self.setup.unet.named_modules():
            if "attn2" in name and "up_blocks.2" in name: 
                handles.append(module.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            self.setup.unet(
                latent.unsqueeze(0), 
                torch.tensor([0], device=self.setup.device),
                encoder_hidden_states=text_embeddings
            )
            
        for h in handles: h.remove()
        
        if not attn_maps: return None

        # 4. Aggregate & Threshold
        avg_map = torch.cat(attn_maps, dim=0).mean(dim=0).mean(dim=0)
        word_map = avg_map[:, word_indices].mean(dim=-1)
        
        res = int(math.sqrt(word_map.shape[0]))
        word_map = word_map.reshape(res, res)
        word_map = (word_map - word_map.min()) / (word_map.max() - word_map.min())
        
        # Object = 1, Background = 0
        object_mask = (word_map > 0.3).float()
        
        # Invert: Background = 1 (Keep), Object = 0 (Change)
        final_mask = 1.0 - object_mask
        final_mask = final_mask.unsqueeze(0).unsqueeze(0).to(self.setup.device)
        
        return final_mask