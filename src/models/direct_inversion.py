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
from diffusers import StableDiffusionPipeline, DDIMScheduler

from .base_editor import BaseEditor
from .p2p.inversion import DirectInversion
from .p2p.attention_control import AttentionStore, make_controller
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
        
        # Force CPU mode if specified (disable MPS completely)
        if device == 'cpu':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
            # Explicitly disable MPS
            if hasattr(torch.backends, 'mps'):
                torch.backends.mps.is_available = lambda: False
        elif isinstance(device, str) and device.startswith("cuda") and os.name != 'nt':
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        print(f"Loading Stable Diffusion model: {model_id}")
        print(f"Device: {device}")
        
        # Initialize DDIM scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        
        # Decide dtype based on device to save GPU memory
        if isinstance(device, str) and device.startswith("cuda"):
            pipeline_dtype = torch.float16
        else:
            pipeline_dtype = torch.float32

        # Load Stable Diffusion model
        # Note: This will download the model on first run (~4GB)
        try:
            self.model = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=self.scheduler,
                torch_dtype=pipeline_dtype,
                safety_checker=None,  # Disable safety checker to save memory
                requires_safety_checker=False,
                low_cpu_mem_usage=True if device == 'cpu' else False,
                use_safetensors=False  # Use original PyTorch format
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            # Try loading without safety checker in a different way
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
            
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            
            # Create pipeline manually
            self.model = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
        
        # Move to device AFTER loading
        self.model = self.model.to(device)

        # Determine low-resource mode for smaller GPUs
        self.low_resource = False
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            try:
                device_obj = torch.device(device)
            except Exception:
                device_obj = torch.device("cuda")
            device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
            try:
                total_mem = torch.cuda.get_device_properties(device_index).total_memory
            except Exception:
                total_mem = None
            if total_mem is not None and total_mem <= 10 * 1024 ** 3:
                self.low_resource = True

        if isinstance(device, str) and device.startswith("cuda"):
            try:
                self.model.enable_attention_slicing()
            except Exception:
                pass
            if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                try:
                    self.model.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            try:
                self.model.enable_vae_slicing()
                self.model.enable_vae_tiling()
            except Exception:
                pass
            if self.low_resource and os.name != 'nt':
                try:
                    self.model.enable_sequential_cpu_offload()
                except Exception:
                    pass
        
        # For CPU, use slower but more stable attention
        if device == 'cpu':
            try:
                self.model.enable_attention_slicing(slice_size=1)  # Reduce memory
            except:
                pass
            # Disable xformers if enabled
            if hasattr(self.model, 'disable_xformers_memory_efficient_attention'):
                try:
                    self.model.disable_xformers_memory_efficient_attention()
                except:
                    pass
        
        try:
            print("Setting timesteps...")
            self.model.scheduler.set_timesteps(self.num_ddim_steps)
            print(f"✓ Timesteps set to {self.num_ddim_steps}")
        except Exception as e:
            print(f"❌ Error setting timesteps: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initialize Direct Inversion
        try:
            print("Initializing Direct Inversion...")
            self.inverter = DirectInversion(
                model=self.model,
                num_ddim_steps=self.num_ddim_steps
            )
            print("✓ Direct Inversion initialized")
        except Exception as e:
            print(f"❌ Error initializing inverter: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("✅ Direct Inversion Editor fully initialized!")
    
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
            generator=None,
            low_resource=self.low_resource
        )
        reconstruct_image = latent2image(vae=self.model.vae, latents=reconstruct_latent)[0]
        if verbose:
            print("Reconstruction complete")
        
        # Step 3: Edit with P2P
        if verbose:
            print("\n[3/3] Performing P2P editing...")
        # Parse blend words if provided
        if blend_word is not None:
            blend_words = [blend_word.split()[0], blend_word.split()[1]]
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
            generator=None,
            low_resource=self.low_resource
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

