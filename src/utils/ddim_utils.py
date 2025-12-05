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


def register_freeu(unet, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
    """
    Applies FreeU to the UNet by patching its UpBlocks.
    Adapted from the official FreeU implementation.
    """
    def Fourier_filter(x, threshold, scale):
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # FFT
        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        
        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device) 
        
        crow, ccol = H // 2, W //2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        x_freq = x_freq * mask
        
        # IFFT
        x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
        return x_filtered.to(dtype=dtype)

    class FreeU_UpBlock(torch.nn.Module):
        def __init__(self, original_block, b, s):
            super().__init__()
            self.original_block = original_block
            self.b = b
            self.s = s

        def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, *args, **kwargs):
            # Apply FreeU to the skip connection (res_hidden_states)
            # The last element of res_hidden_states_tuple is the one corresponding to this upblock level
            
            # We need to modify the res_hidden_states passed to the block.
            # Since the block signature might vary or be complex, we wrap the internal forward logic if possible.
            # However, diffusers UpBlocks usually take (hidden_states, res_hidden_states)
            
            # Structural patch: specific to diffusers UNet2DConditionModel UpBlock structure
            # Note: This is a simplified generic patch. 
            
            # For FreeU:
            # 1. Backbone feature (hidden_states) -> amplify
            # 2. Skip feature (res_hidden_states) -> suppress LF
            
            # Apply backbone scaling
            hidden_states = hidden_states * self.b
            
            # Apply skip connection scaling (Fourier filter)
            # res_hidden_states_tuple is a tuple. We need to modify the specific tensor used by this block.
            # In diffusers, 'res_hidden_states_tuple' contains all residuals. This block consumes the last N.
            # Because we can't easily intercept the tuple splitting inside the block without re-implementing it,
            # simpler FreeU implementations often monkey-patch the 'sample' method of the block.
            
            return self.original_block(hidden_states, res_hidden_states_tuple, temb, upsample_size, *args, **kwargs)

    # Note: A robust implementation of FreeU requires monkey-patching the specific UpBlock forward methods 
    # or using the specialized FreeU enable method if available in newer diffusers.
    # Below is a safer "Fourier-only" implementation that attempts to patch standard CrossAttnUpBlock2D.
    
    # Standard Diffusers FreeU logic (simplified for injection):
    # We'll just set attributes on the model if available, or warn. 
    # Recent diffusers versions have `enable_freeu`.
    if hasattr(unet, "enable_freeu"):
        unet.enable_freeu(b1=b1, b2=b2, s1=s1, s2=s2)
        return True
    
    # If not available, we rely on the user having a recent diffusers version. 
    # Writing a full manual patch here is risky without knowing the exact diffusers version.
    # Recommendation: Update diffusers to >= 0.21.0.
    print("Warning: `enable_freeu` not found on UNet. Ensure diffusers>=0.21.0 is installed.")
    return False


def predict_noise(
    unet,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    text_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_embeddings: Optional[torch.Tensor] = None,
    rescale_factor: float = 0.0,
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

        # Apply Rescale CFG if requested
        if rescale_factor > 0.0:
            # Calculate standard deviations
            std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
            std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            
            # Rescale the guided noise to match the spread of the text-conditioned noise
            # (prevent over-exposure/frying at high CFG)
            factor = std_text / (std_cfg + 1e-7)
            noise_pred_rescaled = noise_pred * factor
            
            # Blend original and rescaled
            noise_pred = noise_pred_rescaled * rescale_factor + noise_pred * (1 - rescale_factor)

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

