"""
Null-Text Inversion implementation.

This module exposes `NullTextEditor`, which extends the standard DDIM editing pipeline by:
1. Performing an initial DDIM inversion to get a latent trajectory.
2. Optimizing the unconditional ("null") embeddings at each timestep to faithfully reconstruct
   the original image from the inverted latents (Null-Text Inversion).
3. Using these optimized embeddings during the editing phase for high-fidelity editing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

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


class NullTextInversion:
    """
    Implements Null-Text Inversion logic.
    Wraps a StandardImageEditingSetup to perform trajectory capture and optimization.
    """

    def __init__(self, setup: StandardImageEditingSetup, config: NullTextConfig):
        self.setup = setup
        self.config = config
        # We use the base DDIMInversion to capture the initial forward trajectory
        self.base_inversion = DDIMInversion(setup, config)

    
    def invert(self, image: Any, source_prompt: str) -> Dict[str, Any]:
        """
        Phase 1 & 2 of Null-Text Inversion:
        1. Capture latent trajectory using standard DDIM Inversion.
        2. Optimize unconditional embeddings to reconstruct that trajectory.
        """
        # 1. Get the reference trajectory (z0 -> zT)
        # Note: Standard Null-Text often uses guidance_scale=1.0 for the inversion trace,
        # but captures the trajectory to then optimize against with higher guidance.
        # We adhere to the config's inversion_guidance_scale (usually 1.0).
        inversion_result = self.base_inversion.invert(image, source_prompt)
        
        latents = inversion_result["latents"]  # [z0, z1, ..., zT]
        text_embeddings = inversion_result["text_embeddings"]
        # Initial null embedding (shared across all steps initially)
        initial_uncond_embeddings = inversion_result["uncond_embeddings"]

        # 2. Optimize Null Embeddings
        # We need to run the generation process (zT -> z0) and optimize 'uncond' at each step.
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
        latents: [z0, z1, ..., zT] obtained from inversion.
        """
        scheduler = self.setup.scheduler
        # Ensure scheduler timesteps are set for inference (T -> 0)
        scheduler.set_timesteps(self.config.num_inference_steps)
        
        # We'll store one optimized embedding per timestep
        optimized_embeddings = []
        
        # We iterate zT -> z0. 
        # Corresponds to latents indices: T, T-1, ..., 1.
        # Target for step starting at z_t is z_{t-1}.
        
        # Make a progress bar that refreshes in place (position=1 for nested bar, leave=False so it doesn't interfere with global progress bar)
        # Use file=sys.stderr explicitly to ensure proper coordination with outer progress bar
        # The key is position=1 (outer bar should be position=0) and leave=False
        # Use total=len() and manual update for better control
        timesteps_list = list(scheduler.timesteps)
        pbar = tqdm(
            total=len(timesteps_list),
            desc="Null-Text Optimization", 
            leave=False, 
            position=1, 
            file=sys.stderr,
            mininterval=0.1,
            maxinterval=1.0,
            disable=False,
            smoothing=0.1,
            dynamic_ncols=False,
            ncols=100
        )
        
        # Current latent starts at zT
        # latents list is [z0, ..., zT], so zT is at index -1
        current_latent_idx = len(latents) - 1
        
        for timestep in timesteps_list:
            # The latent we start from at this step
            zt = latents[current_latent_idx]
            # The latent we want to arrive at (z_{t-1})
            target_latent = latents[current_latent_idx - 1]
            
            # Initialize optimization parameter for this step
            # Clone and enable gradients
            null_embedding = initial_uncond_embeddings.clone().detach()
            null_embedding.requires_grad = True
            
            optimizer = Adam([null_embedding], lr=self.config.null_lr)
            
            # Optimization loop for this timestep
            for _ in range(self.config.null_inner_steps):
                optimizer.zero_grad()
                
                # Predict noise using the current (optimizing) null embedding
                # Note: We must allow gradients to flow through predict_noise -> unet
                # predict_noise helper usually has @torch.no_grad(), so we might need to manually call unet here 
                # or ensure predict_noise context is handled. 
                # The src/utils/ddim_utils.py predict_noise generally assumes no_grad context is external 
                # or is purely inference. Let's manually implement the forward pass to be safe and explicit about gradients.
                
                # Expand latents for CFG
                latent_input = torch.cat([zt] * 2)
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                
                # Concat conditional and current unconditional
                combined_embeddings = torch.cat([null_embedding, text_embeddings])
                
                # UNet forward
                noise_pred = self.setup.unet(
                    latent_input, 
                    timestep, 
                    encoder_hidden_states=combined_embeddings
                ).sample

                # Perform CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # OPTIONAL: Apply Rescale CFG during optimization?
                # If enabled, we should probably apply it here too so the null embeddings 
                # are optimized against the actual physics used in generation.
                if self.config.rescale_factor > 0.0:
                    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
                    std_cfg = noise_pred_cfg.std(dim=list(range(1, noise_pred_cfg.ndim)), keepdim=True)
                    factor = std_text / (std_cfg + 1e-7)
                    noise_pred_cfg = noise_pred_cfg * factor * self.config.rescale_factor + noise_pred_cfg * (1 - self.config.rescale_factor)

                # Predict x_prev (z_{t-1})
                # We reuse ddim_step_forward logic but need it to support gradients.
                # Standard ddim_step_forward usually works with tensors, so it should be autograd-compatible 
                # provided scheduler functions are compliant.
                prev_latent_pred = ddim_step_forward(
                    scheduler, noise_pred_cfg, timestep, zt
                )

                # Calculate Loss (MSE between predicted z_{t-1} and actual z_{t-1} from inversion)
                loss = F.mse_loss(prev_latent_pred, target_latent)
                
                loss.backward()
                optimizer.step()
            
            # Store the optimized embedding (detach to save memory/graph)
            optimized_embeddings.append(null_embedding.detach())
            
            # Move to next step in trajectory
            current_latent_idx -= 1
            
            # Update progress bar manually
            pbar.update(1)
        
        pbar.close()
        return optimized_embeddings

    @torch.no_grad()
    def edit(
        self,
        inversion_result: Dict[str, Any],
        target_prompt: str,
        attention_controller: Optional[PromptToPromptController] = None,
    ) -> torch.Tensor:
        """
        Executes the editing process using the optimized null embeddings.
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
        
        # Retrieve the list of optimized null embeddings
        # They were stored in order of generation (T -> 0), matching scheduler.timesteps
        optimized_uncond_embeddings = inversion_result.get("optimized_uncond_embeddings")
        if optimized_uncond_embeddings is None:
            raise ValueError("Optimized unconditional embeddings not found in inversion result.")

        for idx, timestep in enumerate(scheduler.timesteps):
            if attention_controller is not None:
                attention_controller.set_step(idx)
            
            # Use the specific null embedding for this timestep
            # Default fallback to the global one if index issue (shouldn't happen)
            if idx < len(optimized_uncond_embeddings):
                uncond_embeddings = optimized_uncond_embeddings[idx]
            else:
                uncond_embeddings = inversion_result["uncond_embeddings"]

            noise_pred = predict_noise(
                self.setup.unet,
                zt,
                timestep,
                target_embeddings,
                guidance_scale=self.config.guidance_scale,
                uncond_embeddings=uncond_embeddings,
                rescale_factor=self.config.rescale_factor,
            )
            
            zt = ddim_step_forward(scheduler, noise_pred, timestep, zt)

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
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        resolved_device = self._resolve_device(device)
        dtype = self._resolve_dtype(resolved_device, precision)

        self.config = NullTextConfig()
        
        # Apply standard config args
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
            # For Null-Text, inversion steps typically match inference steps 
            # to map 1:1 for optimization
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
        
        self.config.dtype = dtype
        self.config.update(config_overrides)

        set_seed(self.config.seed)

        self.setup = StandardImageEditingSetup(model_id=model_id, device=resolved_device, dtype=dtype)
        self.inversion = NullTextInversion(self.setup, self.config)
        
        # To reuse DDIMEditor logic for prompt-to-prompt if needed
        # We can implement helper methods similar to DDIMEditor
        self._last_attention_stats: Optional[Dict[str, Any]] = None

    # Helper methods duplicated from DDIMEditor for convenience/independence
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
        
        if verbose:
            print(f"Starting Null-Text Inversion (optimization steps={self.config.null_inner_steps})...")
            
        inversion_result = self.inversion.invert(image, prompt_src)
        
        controller_stats = None
        # Null-Text Inversion is almost always paired with Prompt-to-Prompt (P2P)
        # to preserve spatial structure.
        if self.config.use_prompt_to_prompt:
            edited_tensor, controller_stats = self.apply_prompt_to_prompt(
                inversion_result,
                prompt_src,
                prompt_tar,
                blend_word=blend_word,
            )
        else:
            edited_tensor = self.inversion.edit(inversion_result, prompt_tar)
            
        self._last_attention_stats = controller_stats

        # Reconstruct for comparison (using the optimized nulls, should be very close to original)
        # Note: To strictly verify reconstruction, we'd run edit() with prompt_src. 
        # Here we just decode z0 from the inversion trace for "rough" reconstruction 
        # or we could run a forward pass with optimized nulls.
        # For speed, we just take z0 from inversion result.
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
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Applies Prompt-to-Prompt editing using the optimized null embeddings.
        """
        controller = self._build_prompt_to_prompt_controller(prompt_tar, blend_word)
        if controller is None:
            edited_tensor = self.inversion.edit(inversion_result, prompt_tar)
            return edited_tensor, None

        # P2P typically requires:
        # 1. 'Record' pass: Generate with source prompt (using optimized nulls) to capture attention maps.
        # 2. 'Apply' pass: Generate with target prompt (using optimized nulls), injecting maps from record pass.
        
        with self._use_attention_controller(controller):
            # Record pass
            controller.set_mode("record")
            self.inversion.edit(
                inversion_result,
                prompt_src,
                attention_controller=controller,
            )
            
            # Apply pass
            controller.set_mode("apply")
            edited_tensor = self.inversion.edit(
                inversion_result,
                prompt_tar,
                attention_controller=controller,
            )
            
        return edited_tensor, controller.summary()

    # Reuse the P2P helpers from DDIMEditor logic (duplicated to stay self-contained)
    def _build_prompt_to_prompt_controller(
        self,
        prompt_tar: str,
        blend_word: Optional[str],
    ) -> Optional[PromptToPromptController]:
        if not self.config.use_prompt_to_prompt:
            return None
        # We assume DDIMEditor or ddim.py static methods are not easily importable as mixins, 
        # so we replicate the simple controller builder logic.
        from src.models.ddim import DDIMEditor
        
        # We can actually delegate to a temporary DDIMEditor instance or copy logic.
        # Copying logic is safer to avoid circular init issues.
        controller = PromptToPromptController(
            self_replace_steps=self.config.self_replace_steps,
            cross_replace_steps=self.config.cross_replace_steps,
            store_self_attention=self.config.p2p_store_self_attention,
            attention_dtype=DDIMEditor._resolve_attention_dtype(self.config.p2p_attention_dtype),
            step_stride=max(1, self.config.p2p_attention_step_stride),
            layer_keywords=self.config.p2p_layer_keywords or None,
            max_attention_size=self.config.p2p_max_attention_size,
        )
        if blend_word:
            # Reusing the logic from DDIMEditor via a temporary instance approach is messy.
            # Let's just implement the mask computation locally.
            mask = self._compute_replace_mask(prompt_tar, blend_word)
            if mask is not None:
                controller.set_replace_token_mask(mask)
        return controller

    def _compute_replace_mask(self, prompt: str, blend_word: str) -> Optional[torch.Tensor]:
        # Duplicate of DDIMEditor._compute_replace_mask
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