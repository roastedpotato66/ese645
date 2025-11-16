"""
Direct Inversion baseline implementation.
Combines DDIM inversion with Prompt-to-Prompt editing.

Reference: PIE-Bench paper, Table 1
- Quick and simple editing but less flexible
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

from .base_editor import BaseEditor
from .p2p.inversion import DirectInversion
from .p2p.attention_control import AttentionStore, make_controller
from .p2p.scheduler_dev import DDIMSchedulerDev  # Use author's custom scheduler
from src.utils.image_utils import load_512, latent2image
from src.utils.diffusion_utils import direct_inversion_p2p_guidance_forward


class DirectInversionEditor(BaseEditor):
    """
    Direct Inversion + Prompt-to-Prompt Editor.
    
    This is the simplest baseline method for image editing.
    It performs:
    1. DDIM inversion to convert image to noise
    2. Tracks noise predictions during inversion
    3. Uses P2P attention control during forward diffusion
    4. Applies noise corrections for better reconstruction
    """
    
    def __init__(self, device='cuda', num_ddim_steps=50, model_id="CompVis/stable-diffusion-v1-4"):
        """
        Initialize Direct Inversion editor.
        
        Args:
            device: Device to run on ('cuda', 'mps', or 'cpu')
            num_ddim_steps: Number of DDIM steps
            model_id: HuggingFace model ID for Stable Diffusion
        """
        super().__init__(device=device, num_ddim_steps=num_ddim_steps)
        
        # Initialize DDIM scheduler (using author's custom DDIMSchedulerDev)
        self.scheduler = DDIMSchedulerDev(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
            # Note: Using DDIMSchedulerDev which matches author's diffusers==0.10.0 behavior
        )
        
        # Load Stable Diffusion model (matching author's approach EXACTLY)
        # Author uses simple loading in float32, then .to(device)
        self.model = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=self.scheduler
        ).to(device)
        
        # Set timesteps (matching author)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        
        # Initialize Direct Inversion
        self.inverter = DirectInversion(
            model=self.model,
            num_ddim_steps=self.num_ddim_steps
        )
    
    def edit_image(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        output_path=None,
        return_intermediate=False,
        verbose=True
    ):
        """
        Edit an image using Direct Inversion + P2P.
        
        Args:
            image_path: Path to input image or numpy array
            prompt_src: Source prompt describing the original image
            prompt_tar: Target prompt describing desired edit
            guidance_scale: Classifier-free guidance scale (default: 7.5)
            cross_replace_steps: Fraction of steps to apply cross-attention control
            self_replace_steps: Fraction of steps to apply self-attention control
            blend_word: Words to blend between source and target (format: "word1 word2")
            output_path: Optional path to save output image
            return_intermediate: If True, return intermediate results
            
        Returns:
            PIL.Image or dict: Edited image (or dict with intermediate results)
        """
        # Load and preprocess image
        if verbose:
            print(f"Loading image from: {image_path}")
        image_gt = load_512(image_path)
        
        prompts = [prompt_src, prompt_tar]
        if verbose:
            print(f"Source prompt: {prompt_src}")
            print(f"Target prompt: {prompt_tar}")
        
        # Step 1: Perform Direct Inversion
        if verbose:
            print("\n[1/3] Performing Direct Inversion...")
        image_rec, image_rec_latent, x_stars, noise_loss_list = self.inverter.invert(
            image_gt=image_gt,
            prompt=prompts,
            guidance_scale=guidance_scale,
            verbose=verbose
        )
        x_t = x_stars[-1]  # Get the inverted noise latent
        if verbose:
            print(f"Inversion complete. Latent shape: {x_t.shape}")
        
        # Step 2: Reconstruct with Direct Inversion
        if verbose:
            print("\n[2/3] Reconstructing image...")
        controller = AttentionStore()
        reconstruct_latent, _ = direct_inversion_p2p_guidance_forward(
            model=self.model,
            prompt=prompts,
            controller=controller,
            noise_loss_list=noise_loss_list,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None
        )
        reconstruct_image = latent2image(vae=self.model.vae, latents=reconstruct_latent)[0]
        if verbose:
            print("Reconstruction complete")
        
        # Step 3: Edit with P2P
        if verbose:
            print("\n[3/3] Performing P2P editing...")
        # Parse blend words if provided (match author's format)
        # Author uses: (((word1,), (word2,))) format
        if blend_word is not None and isinstance(blend_word, str) and blend_word.strip():
            blend_word_parts = blend_word.split()
            if len(blend_word_parts) >= 2:
                # Format: (((word_from_source,), (word_from_target,)))
                blend_words = (((blend_word_parts[0],), (blend_word_parts[1],)))
            else:
                blend_words = None  # Invalid blend_word, skip blending
        elif blend_word is not None and not isinstance(blend_word, str):
            # Already in correct format (tuple)
            blend_words = blend_word
        else:
            blend_words = None
        
        # Create P2P controller
        cross_replace_steps_dict = {'default_': cross_replace_steps}
        controller = make_controller(
            pipeline=self.model,
            prompts=prompts,
            is_replace_controller=False,  # Use Refine for smoother edits
            cross_replace_steps=cross_replace_steps_dict,
            self_replace_steps=self_replace_steps,
            blend_words=blend_words,
            num_ddim_steps=self.num_ddim_steps,
            device=self.device
        )
        
        # Run editing with P2P
        edited_latent, _ = direct_inversion_p2p_guidance_forward(
            model=self.model,
            prompt=prompts,
            controller=controller,
            noise_loss_list=noise_loss_list,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None
        )
        
        edited_images = latent2image(vae=self.model.vae, latents=edited_latent)
        edited_image = edited_images[-1]  # Get the edited (target) image
        if verbose:
            print("P2P editing complete")
        
        # Convert to PIL Image
        result_image = Image.fromarray(edited_image)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_image.save(output_path)
            if verbose:
                print(f"\nSaved edited image to: {output_path}")
        
        # Return results
        if return_intermediate:
            return {
                'source': Image.fromarray(image_gt),
                'reconstructed': Image.fromarray(reconstruct_image),
                'edited': result_image,
                'latent': x_t,
                'noise_loss_list': noise_loss_list
            }
        else:
            return result_image


def test_direct_inversion():
    """Test function for Direct Inversion."""
    print("="*60)
    print("Testing Direct Inversion Editor")
    print("="*60)
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Create editor
    # Note: This will download the SD model on first run
    editor = DirectInversionEditor(device=device, num_ddim_steps=10)  # Use fewer steps for testing
    
    print("\nEditor created successfully!")
    print("Next steps:")
    print("1. Implement attention control in src/models/p2p/attention_control.py")
    print("2. Implement forward diffusion in src/utils/diffusion_utils.py")
    print("3. Test on actual images from PIE-Bench dataset")
    print("="*60)


if __name__ == "__main__":
    test_direct_inversion()

