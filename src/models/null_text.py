"""
Null-Text Inversion implementation with ControlNet support.

This module exposes `NullTextEditor`, which extends the standard DDIM editing pipeline by:
1. Performing an initial DDIM inversion to get a latent trajectory.
2. Optimizing the unconditional ("null") embeddings at each timestep to faithfully reconstruct
   the original image from the inverted latents (Null-Text Inversion).
3. Using these optimized embeddings AND ControlNet guidance during the editing phase 
   for high-fidelity structure preservation (optimized for PIE dataset).
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image

# Diffusers imports for ControlNet
from diffusers import ControlNetModel

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
    encode_image,
    encode_prompt,
    get_null_embedding,
    load_image,
    predict_noise,
    set_seed,
    tensor_to_pil,
)
from src.utils.prompt_to_prompt import PromptToPromptController


@dataclass
class NullTextConfig(DDIMConfig):
    """
    Configuration for Null-Text Inversion, extending standard DDIM config.
    """
    # Number of optimization steps for the null text embedding at each timestep
    null_inner_steps: int = 10
    # Learning rate for the null text optimization
    null_lr: float = 0.01
    
    # === ControlNet Configuration ===
    use_controlnet: bool = True
    # Default to Canny for structure preservation (good for PIE dataset)
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    # Low scale (0.3 - 0.5) acts as a "soft constraint" for editing
    controlnet_scale: float = 0.5


class NullTextInversion:
    """
    Implements Null-Text Inversion logic.
    Wraps a StandardImageEditingSetup to perform trajectory capture and optimization.
    """

    def __init__(
        self, 
        setup: StandardImageEditingSetup, 
        config: NullTextConfig,
        controlnet: Optional[ControlNetModel] = None
    ):
        self.setup = setup
        self.config = config
        self.controlnet = controlnet
        # We use the base DDIMInversion to capture the initial forward trajectory
        self.base_inversion = DDIMInversion(setup, config)

    def invert(self, image: Any, source_prompt: str) -> Dict[str, Any]:
        """
        Phase 1 & 2 of Null-Text Inversion:
        1. Capture latent trajectory using standard DDIM Inversion.
        2. Optimize unconditional embeddings to reconstruct that trajectory.
        """
        # 1. Get the reference trajectory (z0 -> zT)
        inversion_result = self.base_inversion.invert(image, source_prompt)
        
        latents = inversion_result["latents"]  # [z0, z1, ..., zT]
        text_embeddings = inversion_result["text_embeddings"]
        # Initial null embedding (shared across all steps initially)
        initial_uncond_embeddings = inversion_result["uncond_embeddings"]

        # 2. Optimize Null Embeddings
        # We run the generation process (zT -> z0) and optimize 'uncond' at each step.
        # Note: We usually do NOT use ControlNet during optimization phase, 
        # as we want the null embeddings to learn the reconstruction purely.
        optimized_uncond_embeddings = self.optimize_null_text(
            latents, 
            text_embeddings, 
            initial_uncond_embeddings
        )

        # Update result with the optimized list
        inversion_result["optimized_uncond_embeddings"] = optimized_uncond_embeddings
        return inversion_result

    def optimize_null_text(
        self, 
        latents: List[torch.Tensor], 
        text_embeddings: torch.Tensor, 
        initial_uncond_embeddings: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Optimizes the unconditional embeddings to match the latent trajectory.
        """
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
                
                noise_pred = self.setup.unet(
                    latent_input, 
                    timestep, 
                    encoder_hidden_states=combined_embeddings
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                prev_latent_pred = ddim_step_forward(
                    scheduler, noise_pred_cfg, timestep, current_latent
                )

                loss = F.mse_loss(prev_latent_pred, target_latent)
                
                loss.backward()
                optimizer.step()
           
            optimized_embeddings.append(current_null_embedding.detach().clone())
            
            # Reconstruction step for next iteration
            with torch.no_grad():
                latent_input = torch.cat([current_latent] * 2)
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                combined_embeddings = torch.cat([current_null_embedding, text_embeddings])
                
                noise_pred = self.setup.unet(
                    latent_input, 
                    timestep, 
                    encoder_hidden_states=combined_embeddings
                ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                current_latent = ddim_step_forward(
                    scheduler, noise_pred_cfg, timestep, current_latent
                )
            
            current_null_embedding = current_null_embedding.detach()
            current_null_embedding.requires_grad = True
            
            pbar.update(1)
        
        pbar.close()
        return optimized_embeddings

    @torch.no_grad()
    def edit(
        self,
        inversion_result: Dict[str, Any],
        target_prompt: str,
        attention_controller: Optional[PromptToPromptController] = None,
        control_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executes the editing process using the optimized null embeddings AND ControlNet.
        """
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Prepare controller
        total_steps = len(scheduler.timesteps)
        if attention_controller is not None:
            attention_controller.set_total_steps(total_steps)

        # Start from zT
        zt = inversion_result["latents"][-1]
        
        target_embeddings = encode_prompt(
            self.setup.text_encoder,
            self.setup.tokenizer,
            target_prompt,
            self.setup.device,
        )
        
        optimized_uncond_embeddings = inversion_result.get("optimized_uncond_embeddings")
        if optimized_uncond_embeddings is None:
            raise ValueError("Optimized unconditional embeddings not found in inversion result.")

        for idx, timestep in enumerate(scheduler.timesteps):
            if attention_controller is not None:
                attention_controller.set_step(idx)
            
            # Use the specific null embedding for this timestep
            if idx < len(optimized_uncond_embeddings):
                uncond_embeddings = optimized_uncond_embeddings[idx]
            else:
                uncond_embeddings = inversion_result["uncond_embeddings"]

            # Prepare inputs for ControlNet and UNet
            # We need to construct the combined embeddings: [Uncond, Text]
            combined_embeddings = torch.cat([uncond_embeddings, target_embeddings])
            
            latent_input = torch.cat([zt] * 2)
            latent_input = scheduler.scale_model_input(latent_input, timestep)
            
            # === ControlNet Inference ===
            down_block_res_samples = None
            mid_block_res_sample = None
            
            if self.controlnet is not None and control_image is not None:
                # ControlNet forward pass
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=combined_embeddings,
                    controlnet_cond=control_image,
                    conditioning_scale=self.config.controlnet_scale,
                    return_dict=False,
                )

            # === UNet Inference ===
            # Inject ControlNet residuals into UNet
            noise_pred = self.setup.unet(
                latent_input, 
                timestep, 
                encoder_hidden_states=combined_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            # Standard CFG guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            zt = ddim_step_forward(scheduler, noise_pred_cfg, timestep, zt)

        decoded = decode_latent(self.setup.vae, zt, self.setup.device)
        return decoded


@register_model("null_text")
class NullTextEditor:
    """
    High-level interface for Null-Text Inversion editing.
    Compatible with the standardized test scripts.
    """

    def __init__(
        self,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        num_inference_steps: Optional[int] = None,
        num_inversion_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        precision: Optional[str] = None,
        # Null-Text specific
        null_inner_steps: int = 10,
        null_lr: float = 1e-2,
        # ControlNet specific
        use_controlnet: bool = True,
        controlnet_scale: float = 0.5,
        controlnet_model_id: str = "lllyasviel/sd-controlnet-canny", # <--- 新增这一行
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        resolved_device = self._resolve_device(device)
        dtype = self._resolve_dtype(resolved_device, precision)

        self.config = NullTextConfig()
        
        # Apply standard config args
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
            if num_inversion_steps is None:
                self.config.num_inversion_steps = num_inference_steps
        
        if num_inversion_steps is not None:
            self.config.num_inversion_steps = num_inversion_steps
            
        if guidance_scale is not None:
            self.config.guidance_scale = guidance_scale
        if seed is not None:
            self.config.seed = seed
        
        # Apply Null-Text specific args
        self.config.null_inner_steps = null_inner_steps
        self.config.null_lr = null_lr
        
        # Apply ControlNet specific args
        self.config.use_controlnet = use_controlnet
        self.config.controlnet_scale = controlnet_scale
        self.config.controlnet_model_id = controlnet_model_id # <--- 记得在这里赋值
        
        self.config.dtype = dtype
        self.config.update(config_overrides)

        set_seed(self.config.seed)

        self.setup = StandardImageEditingSetup(model_id=model_id, device=resolved_device, dtype=dtype)
        
        # Initialize ControlNet if requested
        self.controlnet = None
        if self.config.use_controlnet:
            print(f"Loading ControlNet: {self.config.controlnet_model_id}...")
            self.controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model_id, 
                torch_dtype=dtype
            ).to(resolved_device)
        
        self.inversion = NullTextInversion(self.setup, self.config, controlnet=self.controlnet)
        
        self._last_attention_stats: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(device: torch.device, precision: Optional[str]) -> torch.dtype:
        if precision is not None:
            if precision == "fp32":
                return torch.float32
            if precision == "fp16":
                return torch.float16
        if device.type == "cpu":
            return torch.float32
        return torch.float32

    def prepare_control_image(self, image_pil: Image.Image) -> torch.Tensor:
        """
        Preprocesses the input image for ControlNet (default: Canny edge detection).
        Returns a tensor ready for the ControlNet forward pass.
        """
        if not self.config.use_controlnet:
            return None
            
        image = np.array(image_pil)
        
        # Canny Edge Detection
        # Thresholds can be tuned: 100/200 is standard for SD Canny
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        # Convert to tensor [1, 3, H, W] normalized to [0, 1]
        control_image = torch.from_numpy(np.array(image).transpose(2, 0, 1) / 255.0).float()
        control_image = control_image.unsqueeze(0).to(self.setup.device, dtype=self.config.dtype)
        
        return control_image

    def edit_image(
        self,
        image_path: str,
        prompt_src: str,
        prompt_tar: str,
        blend_word: Optional[str] = None,
        output_path: Optional[str] = None,
        return_intermediate: bool = False,
        verbose: bool = False,
        **_: Any,
    ):
        """
        Perform Null-Text inversion/editing on a single image path.
        """
        image = load_image(image_path, self.config.image_size)
        
        # Prepare Control Image (e.g., Canny map)
        control_image = None
        if self.config.use_controlnet:
            control_image = self.prepare_control_image(image)
        
        if verbose:
            print(f"Starting Null-Text Inversion (opt_steps={self.config.null_inner_steps}, controlnet={self.config.use_controlnet})...")
            
        inversion_result = self.inversion.invert(image, prompt_src)
        
        controller_stats = None
        
        if self.config.use_prompt_to_prompt:
            edited_tensor, controller_stats = self.apply_prompt_to_prompt(
                inversion_result,
                prompt_src,
                prompt_tar,
                blend_word=blend_word,
                control_image=control_image
            )
        else:
            edited_tensor = self.inversion.edit(
                inversion_result, 
                prompt_tar,
                control_image=control_image
            )
            
        self._last_attention_stats = controller_stats

        # Reconstruct for comparison
        reconstructed_tensor = decode_latent(
            self.setup.vae, inversion_result["latents"][0], self.setup.device
        )

        source_pil = image
        reconstructed_pil = tensor_to_pil(reconstructed_tensor)
        edited_pil = tensor_to_pil(edited_tensor)

        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            edited_pil.save(output_path)

        if verbose:
            print("Null-Text edit complete.")
            print(f"  Source prompt: {prompt_src}")
            print(f"  Target prompt: {prompt_tar}")
            print(f"  Output path: {output_path}")

        if return_intermediate:
            return {
                "source": source_pil,
                "reconstructed": reconstructed_pil,
                "edited": edited_pil,
            }
        return edited_pil

    def apply_prompt_to_prompt(
        self,
        inversion_result: Dict[str, Any],
        prompt_src: str,
        prompt_tar: str,
        blend_word: Optional[str] = None,
        control_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Applies Prompt-to-Prompt editing using the optimized null embeddings.
        Also accepts control_image to pass down to edit()
        """
        controller = self._build_prompt_to_prompt_controller(prompt_tar, blend_word)
        if controller is None:
            edited_tensor = self.inversion.edit(
                inversion_result, 
                prompt_tar,
                control_image=control_image
            )
            return edited_tensor, None

        with self._use_attention_controller(controller):
            # Record pass:
            # We use the source prompt. We typically also use ControlNet here 
            # to ensure the attention maps align with the structure perfectly.
            controller.set_mode("record")
            self.inversion.edit(
                inversion_result,
                prompt_src,
                attention_controller=controller,
                control_image=control_image,
            )
            
            # Apply pass:
            # We use the target prompt + ControlNet + P2P injection.
            controller.set_mode("apply")
            edited_tensor = self.inversion.edit(
                inversion_result,
                prompt_tar,
                attention_controller=controller,
                control_image=control_image,
            )
            
        return edited_tensor, controller.summary()

    def _build_prompt_to_prompt_controller(
        self,
        prompt_tar: str,
        blend_word: Optional[str],
    ) -> Optional[PromptToPromptController]:
        if not self.config.use_prompt_to_prompt:
            return None

        # Replicate basic controller construction
        controller = PromptToPromptController(
            self_replace_steps=self.config.self_replace_steps,
            cross_replace_steps=self.config.cross_replace_steps,
            store_self_attention=self.config.p2p_store_self_attention,
            # Handle static method resolution (assuming DDIMEditor or ddim module available or replicating resolve logic)
            attention_dtype=torch.float32, # Simplified: force float32 or inspect device
            step_stride=max(1, self.config.p2p_attention_step_stride),
            layer_keywords=self.config.p2p_layer_keywords or None,
            max_attention_size=self.config.p2p_max_attention_size,
        )
        if blend_word:
            mask = self._compute_replace_mask(prompt_tar, blend_word)
            if mask is not None:
                controller.set_replace_token_mask(mask)
        return controller

    def _compute_replace_mask(self, prompt: str, blend_word: str) -> Optional[torch.Tensor]:
        words = [
            w.strip().lower()
            for w in blend_word.replace(",", " ").split()
            if w.strip() and w.strip().lower() != "none"
        ]
        if not words:
            return None
        encoding = self.setup.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.setup.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = encoding.input_ids[0]
        tokens = self.setup.tokenizer.convert_ids_to_tokens(token_ids)
        mask = torch.ones(1, 1, len(tokens), dtype=torch.float32)
        protected = set(words)
        for idx, token in enumerate(tokens):
            cleaned = token.replace("Ġ", "").replace("</w>", "").lower()
            if cleaned in protected:
                mask[0, 0, idx] = 0.0
        return mask

    @contextmanager
    def _use_attention_controller(self, controller: PromptToPromptController):
        if controller is None:
            yield
            return
        original_processors = dict(self.setup.unet.attn_processors)
        processors = controller.build_processors(self.setup.unet)
        self.setup.unet.set_attn_processor(processors)
        try:
            yield
        finally:
            self.setup.unet.set_attn_processor(original_processors)