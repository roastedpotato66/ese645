"""
Helper functions for standardized DDIM inversion/editing.

Implements the universal utilities described in `ddim_overview.md` such as:
- Image ↔ latent conversions
- Prompt encoding helpers
- Deterministic DDIM step functions
- Noise prediction with classifier-free guidance
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image(image_path: str, image_size: int) -> Image.Image:
    """
    Load an image from disk, convert to RGB, and resize to the configured size.
    """
    image = Image.open(image_path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return image


def _to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)


def encode_image(vae, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Encode an RGB image into latent space using the VAE encoder.
    """
    image_tensor = _to_tensor(image).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Decode latent tensor back into an image tensor in [0, 1].
    """
    with torch.no_grad():
        latent = latent / vae.config.scaling_factor
        image = vae.decode(latent).sample
    image = (image + 1.0) / 2.0
    return image.clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a BCHW image tensor in [0, 1] to a PIL image.
    """
    tensor = tensor.detach().cpu().permute(0, 2, 3, 1)[0]
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def encode_prompt(text_encoder, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """
    Tokenize and encode a text prompt to hidden states.
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    return text_embeddings


def get_null_embedding(text_encoder, tokenizer, device: torch.device) -> torch.Tensor:
    """
    Return the unconditional (empty prompt) embedding.
    """
    return encode_prompt(text_encoder, tokenizer, "", device)


def predict_noise(
    unet,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    text_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Predict noise using classifier-free guidance.
    """
    if guidance_scale == 1.0 or uncond_embeddings is None:
        noise_pred = unet(latent, timestep, encoder_hidden_states=text_embeddings).sample
    else:
        latent_input = torch.cat([latent] * 2)
        embeddings = torch.cat([uncond_embeddings, text_embeddings])
        noise_pred = unet(
            latent_input,
            timestep,
            encoder_hidden_states=embeddings,
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def _timesteps_to_index(timestep: torch.Tensor | int) -> int:
    if isinstance(timestep, torch.Tensor):
        return int(timestep.item())
    return int(timestep)


def ddim_step_reverse(
    scheduler,
    model_output: torch.Tensor,
    t_from: torch.Tensor,  # Lower timestep (current)
    t_to: torch.Tensor,    # Higher timestep (target)
    sample: torch.Tensor
) -> torch.Tensor:
    """
    DDIM inversion: go from t_from to t_to (t_to > t_from).
    """
    t_from_idx = _timesteps_to_index(t_from)
    t_to_idx = _timesteps_to_index(t_to)
    
    alpha_from = scheduler.alphas_cumprod[t_from_idx]
    alpha_to = scheduler.alphas_cumprod[t_to_idx]
    
    # Predict x0 from current state
    pred_x0 = (sample - torch.sqrt(1 - alpha_from) * model_output) / torch.sqrt(alpha_from)
    
    # Move to next (noisier) state
    next_sample = torch.sqrt(alpha_to) * pred_x0 + torch.sqrt(1 - alpha_to) * model_output
    
    return next_sample


def ddim_step_forward(scheduler, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """
    Deterministic DDIM sampling step (noise -> image).
    """
    step_output = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=0.0,
        return_dict=False,
    )
    return step_output[0]


def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

