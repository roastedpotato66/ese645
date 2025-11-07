"""
Utility functions for image processing.
Extracted and adapted from PnPInversion project.
"""

import numpy as np
import PIL.Image as Image
import torch


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    """
    Load and resize image to 512x512.
    
    Args:
        image_path: Path to image file or numpy array
        left, right, top, bottom: Crop parameters
        
    Returns:
        numpy.ndarray: Image array (512, 512, 3)
    """
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    
    h, w, c = image.shape
    # Center crop to square
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def image2latent(vae, image):
    """
    Convert image to latent representation using VAE encoder.
    
    Args:
        vae: VAE model from Stable Diffusion
        image: Image (PIL.Image or numpy array)
        
    Returns:
        torch.Tensor: Latent representation
    """
    with torch.no_grad():
        if type(image) is Image.Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # Normalize to [-1, 1]
            image = torch.from_numpy(image).float() / 127.5 - 1
            vae_dtype = getattr(vae, "dtype", torch.float32)
            image = image.permute(2, 0, 1).unsqueeze(0).to(device=vae.device, dtype=vae_dtype)
            
            # Encode to latent
            latents = vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215  # Scaling factor
    
    return latents


@torch.no_grad()
def latent2image(vae, latents, return_type='np'):
    """
    Convert latent representation back to image using VAE decoder.
    
    Args:
        vae: VAE model from Stable Diffusion
        latents: Latent tensor
        return_type: 'np' for numpy array, 'pt' for torch tensor
        
    Returns:
        numpy.ndarray or torch.Tensor: Decoded image
    """
    latents = 1 / 0.18215 * latents.detach()
    latents = latents.to(dtype=getattr(vae, "dtype", latents.dtype))
    image = vae.decode(latents)['sample']
    
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    
    return image


def slerp(val, low, high):
    """
    Spherical linear interpolation.
    
    Args:
        val: Interpolation factor (0 to 1)
        low: Start tensor
        high: End tensor
        
    Returns:
        torch.Tensor: Interpolated tensor
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + \
          (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    """
    Spherical linear interpolation for tensors with any shape.
    
    Args:
        val: Interpolation factor
        low: Start tensor
        high: End tensor
        
    Returns:
        torch.Tensor: Interpolated tensor
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)


def setup_seed(seed=1234):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_latent(latent, model, height, width, generator, batch_size):
    """Initialize latent tensor."""
    channels = getattr(model.unet, "config", model.unet).in_channels
    latent_device = model.device
    latent_dtype = model.unet.dtype

    if latent is None:
        latent = torch.randn(
            (1, channels, height // 8, width // 8),
            generator=generator,
            device=latent_device,
            dtype=latent_dtype,
        )
    else:
        latent = latent.to(device=latent_device, dtype=latent_dtype)

    latents = latent.expand(batch_size, channels, height // 8, width // 8).to(device=latent_device, dtype=latent_dtype)
    return latent, latents


def get_word_inds(text, word_place, tokenizer):
    """Get word indices from tokenizer."""
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds, prompt_ind, word_inds=None):
    """Update alpha for time and word."""
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer, max_num_words=77):
    """Get attention alpha for time and words."""
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


if __name__ == "__main__":
    # Simple test
    print("Testing image utils...")
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"Created dummy image: {dummy_img.shape}")
    
    # Test load function
    processed = load_512(dummy_img)
    print(f"Processed image: {processed.shape}")
    
    print("Image utils test completed!")

