"""
Forward diffusion utilities for Direct Inversion and P2P editing.
"""

import torch
from src.models.p2p.attention_control import register_attention_control
from src.utils.image_utils import init_latent


def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    """Single diffusion step with P2P guidance."""
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    latent=None,
    uncond_embeddings=None
):
    """
    P2P guidance forward diffusion.
    
    Args:
        model: Stable Diffusion model
        prompt: List of prompts
        controller: Attention controller
        num_inference_steps: Number of steps
        guidance_scale: CFG scale
        generator: Random generator
        latent: Starting latent
        uncond_embeddings: Unconditional embeddings
        
    Returns:
        tuple: (final_latent, initial_latent)
    """
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False, add_offset=True):
    """Single diffusion step with Direct Inversion noise correction."""
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    
    # Direct Inversion: Add noise correction for source branch
    if add_offset:
        latents = torch.concat((latents[:1] + noise_loss[:1], latents[1:]))
    
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    noise_loss_list=None,
    add_offset=True
):
    """
    Direct Inversion forward diffusion with noise correction.
    
    This is the key function for Direct Inversion. It uses the noise_loss_list
    tracked during inversion to correct the forward diffusion process.
    
    Args:
        model: Stable Diffusion model
        prompt: List of [source_prompt, target_prompt]
        controller: Attention controller for P2P editing
        latent: Starting noise latent from inversion
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        generator: Random generator (unused but kept for compatibility)
        noise_loss_list: List of noise corrections from inversion
        add_offset: Whether to apply noise correction
        
    Returns:
        tuple: (final_latent, initial_latent)
    """
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    # Encode prompts
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    # Encode unconditional prompt
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # Initialize latent
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    
    # Forward diffusion with noise correction
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step(
            model, controller, latents, context, t, guidance_scale, 
            noise_loss_list[i], low_resource=False, add_offset=add_offset
        )
        
    return latents, latent
