# src/models/direct.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from PIL import Image

from src.models.registry import register_model
from src.models.ddim import (
    DEFAULT_MODEL_ID,
    DDIMConfig,
    StandardImageEditingSetup,
    DDIMInversion,                 # we’ll reuse its *edit* method
)
from src.utils.ddim_utils import (
    encode_image,
    decode_latent,
    encode_prompt,
    get_null_embedding,
    tensor_to_pil,
    load_image,
    set_seed,
)

class _DirectInversion(DDIMInversion):
    """Overrides only the invert() function: one-shot VAE encode."""
    @torch.no_grad()
    def invert(self, image: Image.Image, source_prompt: str) -> Dict[str, Any]:
        z0 = encode_image(self.setup.vae, image, self.setup.device, self.setup.dtype)
        text_emb = encode_prompt(
            self.setup.text_encoder, self.setup.tokenizer,
            source_prompt, self.setup.device
        )
        uncond = get_null_embedding(
            self.setup.text_encoder, self.setup.tokenizer, self.setup.device
        )
        # we store a one-element list so downstream code still sees “latents[0]”
        return {
            "latents": [z0],
            "text_embeddings": text_emb,
            "uncond_embeddings": uncond,
            "num_inversion_steps": 1,
        }

@register_model("direct")
class DirectEditor:
    """
    High-level wrapper just like DDIMEditor, but uses _DirectInversion.
    Call exactly the same test scripts with --model direct
    """
    def __init__(
        self,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        num_inference_steps: int | None = None,
        num_inversion_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        precision: str | None = None,
        config_overrides: Dict[str, Any] | None = None,
    ):
        # resolve device / dtype exactly like DDIMEditor
        from src.models.ddim import DDIMEditor  # local import avoids circular ref
        self._resolve_device = DDIMEditor._resolve_device
        self._resolve_dtype  = DDIMEditor._resolve_dtype
        resolved_device = self._resolve_device(device)
        dtype = self._resolve_dtype(resolved_device, precision)

        self.config = DDIMConfig()
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            self.config.guidance_scale = guidance_scale
        if seed is not None:
            self.config.seed = seed
        self.config.dtype = dtype
        self.config.update(config_overrides)

        set_seed(self.config.seed)

        self.setup     = StandardImageEditingSetup(model_id, resolved_device, dtype)
        self.inversion = _DirectInversion(self.setup, self.config)

    # identical public API to DDIMEditor
    def edit_image(
        self,
        image_path: str,
        prompt_src: str,
        prompt_tar: str,
        output_path: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        image = load_image(image_path, self.config.image_size)
        inv   = self.inversion.invert(image, prompt_src)
        edited = self.inversion.edit(inv, prompt_tar)

        edited_pil = tensor_to_pil(edited)
        if output_path:
            edited_pil.save(output_path)
        if verbose:
            print("Direct-Inversion edit complete →", output_path)
        return edited_pil
