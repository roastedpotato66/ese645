"""
DDIM Inversion methods for diffusion-based image editing.
Extracted and adapted from PnPInversion project.

Implements:
- DirectInversion: Simple DDIM inversion with noise prediction tracking
"""

import torch
import numpy as np
from PIL import Image


class DirectInversion:
    """
    Direct DDIM Inversion method.
    
    This is the simplest baseline method that:
    1. Converts image to latent space using VAE
    2. Performs DDIM inversion to get noise latent
    3. Tracks noise predictions during inversion for later use
    
    Reference: PIE-Bench paper, Table 1
    """
    
    def __init__(self, model, num_ddim_steps=50):
        """
        Initialize Direct Inversion.
        
        Args:
            model: StableDiffusionPipeline model
            num_ddim_steps: Number of DDIM steps
        """
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.tokenizer = model.tokenizer
        self.prompt = None
        self.context = None  # Text embeddings
    
    @property
    def scheduler(self):
        """Access the model's scheduler."""
        return self.model.scheduler
    
    def prev_step(self, model_output, timestep, sample):
        """
        Compute previous latent from current latent (backward step).
        
        Args:
            model_output: Predicted noise from UNet
            timestep: Current timestep
            sample: Current latent
            
        Returns:
            tuple: (prev_sample, difference_scale)
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict original sample
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # Compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        # Compute previous sample
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Compute difference scale (for noise tracking)
        difference_scale_pred_original_sample = - beta_prod_t ** 0.5 / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction

        target_dtype = model_output.dtype
        prev_sample = prev_sample.to(dtype=target_dtype)
        difference_scale = difference_scale.to(dtype=target_dtype)
        
        return prev_sample, difference_scale
    
    def next_step(self, model_output, timestep, sample):
        """
        Compute next latent from current latent (forward step).
        
        Args:
            model_output: Predicted noise from UNet
            timestep: Current timestep
            sample: Current latent
            
        Returns:
            torch.Tensor: Next latent
        """
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 
            999
        ), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        next_sample = next_sample.to(dtype=model_output.dtype)
        
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        """
        Get noise prediction from UNet.
        
        Args:
            latents: Current latent
            t: Timestep
            context: Text embeddings
            
        Returns:
            torch.Tensor: Predicted noise
        """
        latents = latents.to(dtype=self.model.unet.dtype)
        context = context.to(dtype=self.model.unet.dtype)
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    @torch.no_grad()
    def init_prompt(self, prompt):
        """
        Initialize text embeddings for prompts.
        
        Args:
            prompt: List of prompts [source_prompt, target_prompt]
        """
        # Encode unconditional (empty) prompt
        uncond_input = self.model.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        uncond_embeddings = uncond_embeddings.to(dtype=self.model.unet.dtype)
        
        # Encode conditional prompt
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        text_embeddings = text_embeddings.to(dtype=self.model.unet.dtype)
        
        # Concatenate unconditional and conditional embeddings
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
    
    @torch.no_grad()
    def ddim_loop(self, latent):
        """
        Perform DDIM inversion loop: latent -> noise.
        
        Args:
            latent: Initial latent from VAE encoder
            
        Returns:
            list: All latents from inversion process
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        cond_embeddings = cond_embeddings[[0]]  # Use only source prompt
        
        all_latent = [latent]
        latent = latent.clone().detach()
        
        print("Running DDIM Inversion...")
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        
        return all_latent
    
    @torch.no_grad()
    def ddim_inversion(self, image):
        """
        Complete DDIM inversion process.
        
        Args:
            image: Input image (numpy array, 512x512x3)
            
        Returns:
            tuple: (reconstructed_image, inverted_latents, original_latent)
        """
        # Convert image to latent
        from src.utils.image_utils import image2latent, latent2image
        
        latent = image2latent(self.model.vae, image).to(dtype=self.model.unet.dtype)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        
        return image_rec, ddim_latents, latent
    
    def invert(self, image_gt, prompt, guidance_scale=7.5):
        """
        Main inversion method with noise tracking.
        
        Args:
            image_gt: Ground truth image (numpy array, 512x512x3)
            prompt: List of [source_prompt, target_prompt]
            guidance_scale: Guidance scale for classifier-free guidance
            
        Returns:
            tuple: (rec_image, rec_latent, all_latents, noise_loss_list)
        """
        self.init_prompt(prompt)
        
        # Register attention control (placeholder - will be implemented later)
        # register_attention_control(self.model, None)
        
        # Perform inversion
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        
        # Track noise predictions during inversion
        # This is used for Direct Inversion's key feature
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        noise_loss_list = []
        
        latent = image_rec_latent
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            
            # Get noise predictions for both source and target
            cond_src = cond_embeddings[[0]]
            cond_tar = cond_embeddings[[1]]
            
            noise_pred_src = self.get_noise_pred_single(latent, t, cond_src)
            noise_pred_tar = self.get_noise_pred_single(latent, t, cond_tar)
            
            # Compute noise difference and scale
            latent, difference_scale = self.prev_step(noise_pred_src, t, latent)
            
            # Store noise loss for later use
            noise_loss = noise_pred_tar - noise_pred_src
            noise_loss_list.append((noise_loss * difference_scale).to(dtype=self.model.unet.dtype))
        
        return image_rec, image_rec_latent, ddim_latents, noise_loss_list


# Placeholder for other inversion methods that teammates might implement
class NullInversion:
    """Placeholder for Null-text Inversion (to be implemented by teammates)."""
    pass


class NegativePromptInversion:
    """Placeholder for Negative Prompt Inversion (to be implemented by teammates)."""
    pass

