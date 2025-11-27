"""
Direct Inversion implementation.

Direct Inversion (https://arxiv.org/abs/2310.01506) disentangles the source and target branches.
For the source branch, instead of re-computing the diffusion steps (which leads to deviation),
it uses the exact latents obtained during inversion. This ensures perfect content preservation
in the attention maps used for Prompt-to-Prompt editing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from src.models.ddim import DDIMEditor, DDIMInversion
from src.models.registry import register_model
from src.utils.ddim_utils import (
    decode_latent,
    encode_prompt,
    predict_noise,
)
from src.utils.prompt_to_prompt import PromptToPromptController


class DirectInversionImpl(DDIMInversion):
    """
    Extends DDIMInversion to support 'path-forcing' during the source branch reconstruction.
    """

    @torch.no_grad()
    def reconstruct_with_path(
        self,
        inversion_result: Dict[str, Any],
        source_prompt: str,
        attention_controller: Optional[PromptToPromptController] = None,
    ) -> torch.Tensor:
        """
        Runs the generation process for the source prompt, but instead of computing
        the next latent via DDIM, it forces the latent to match the trajectory 
        recorded during inversion.
        """
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        
        # stored latents are [z0, z_t1, z_t2, ..., z_T]
        stored_latents = inversion_result["latents"]
        
        total_steps = len(scheduler.timesteps)
        if attention_controller is not None:
            attention_controller.set_total_steps(total_steps)

        # Start at z_T (the last stored latent)
        zt = stored_latents[-1]
        
        text_embeddings = encode_prompt(
            self.setup.text_encoder,
            self.setup.tokenizer,
            source_prompt,
            self.setup.device,
        )
        uncond_embeddings = inversion_result["uncond_embeddings"]

        for idx, timestep in enumerate(scheduler.timesteps):
            if attention_controller is not None:
                attention_controller.set_step(idx)
            
            # CRITICAL FIX:
            # We must use the *inference* guidance_scale (e.g. 7.5) here, not the inversion scale (1.0).
            # This ensures predict_noise runs with Batch Size = 2 (Uncond + Cond), matching the
            # batch structure that will be used in the 'Apply' (Target) phase.
            # If we used scale=1.0, we'd get Batch Size = 1, causing a shape mismatch in PromptToPrompt.
            # We discard the actual noise prediction anyway, so the scale value doesn't corrupt the path.
            _ = predict_noise(
                self.setup.unet,
                zt,
                timestep,
                text_embeddings,
                guidance_scale=self.config.guidance_scale, 
                uncond_embeddings=uncond_embeddings,
                rescale_factor=self.config.rescale_factor,
            )
            
            # Force the next latent to be the one from the inversion chain
            # stored_latents indices: 0=z0, -1=zT, -2=z_{T-1}
            # At idx=0 (start), we are at zT (-1). We want to go to z_{T-1} (-2).
            next_latent_idx = -(idx + 2)
            
            # Ensure we don't go out of bounds (e.g. past z0)
            if abs(next_latent_idx) <= len(stored_latents):
                zt = stored_latents[next_latent_idx]
            else:
                # Fallback: if steps mismatch, keep current (or break)
                pass

        decoded = decode_latent(self.setup.vae, zt, self.setup.device)
        return decoded


@register_model("direct_inversion")
class DirectInversionEditor(DDIMEditor):
    """
    Direct Inversion Editor.
    
    Uses the standard DDIM Setup but modifies the editing pipeline to use Direct Inversion
    (forcing source latents) during the Prompt-to-Prompt recording phase.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Swap the implementation with our Direct Inversion logic
        self.inversion = DirectInversionImpl(self.setup, self.config)

    def apply_prompt_to_prompt(
        self,
        inversion_result: Dict[str, Any],
        prompt_src: str,
        prompt_tar: str,
        blend_word: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Applies Prompt-to-Prompt editing using Direct Inversion strategy.
        """
        controller = self._build_prompt_to_prompt_controller(prompt_tar, blend_word)
        
        # If P2P is disabled, fallback to standard DDIM edit
        if controller is None:
            edited_tensor = self.inversion.edit(inversion_result, prompt_tar)
            return edited_tensor, None

        with self._use_attention_controller(controller):
            # 1. Source Branch (Record)
            # Direct Inversion Key: Use reconstruct_with_path to enforce exact latents
            controller.set_mode("record")
            self.inversion.reconstruct_with_path(
                inversion_result,
                prompt_src,
                attention_controller=controller,
            )
            
            # 2. Target Branch (Apply)
            # Standard DDIM forward pass for the edited image
            controller.set_mode("apply")
            edited_tensor = self.inversion.edit(
                inversion_result,
                prompt_tar,
                attention_controller=controller,
            )
            
        return edited_tensor, controller.summary()