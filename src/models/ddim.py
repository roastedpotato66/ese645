"""
Standardized DDIM inversion/editing implementation based on `ddim_overview.md`.

This module exposes `DDIMEditor`, a lightweight wrapper that:
- Loads the standardized Stable Diffusion components (VAE/UNet/Text encoder/Scheduler)
- Performs DDIM inversion to capture noise trajectory
- Runs DDIM sampling with classifier-free guidance to produce edited images
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from src.models.registry import register_model
from src.utils.ddim_utils import (
    decode_latent,
    ddim_step_forward,
    ddim_step_reverse,
    encode_image,
    encode_prompt,
    get_null_embedding,
    load_image,
    predict_noise,
    set_seed,
    tensor_to_pil,
)
from src.utils.prompt_to_prompt import PromptToPromptController


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@dataclass
class DDIMConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    inversion_guidance_scale: float = 1.0
    image_size: int = 512
    dtype: torch.dtype = torch.float16
    seed: Optional[int] = 42
    use_prompt_to_prompt: bool = True
    self_replace_steps: float = 0.6
    cross_replace_steps: float = 0.4
    p2p_attention_dtype: str = "float16"
    p2p_store_self_attention: bool = True
    p2p_attention_step_stride: int = 4
    p2p_layer_keywords: List[str] = field(default_factory=list)

    def update(self, overrides: Optional[Dict[str, Any]] = None):
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)


class StandardImageEditingSetup:
    """
    Container for the standardized Stable Diffusion components.
    """

    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            clip_sample=False,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae.eval()

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.unet.eval()

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
        )


class DDIMInversion:
    """
    Implements DDIM inversion (image → latent trajectory) and editing (latent → edited image).
    """

    def __init__(self, setup: StandardImageEditingSetup, config: DDIMConfig):
        self.setup = setup
        self.config = config

    @torch.no_grad()
    def invert(self, image: Image.Image, source_prompt: str) -> Dict[str, Any]:
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        
        # DON'T flip! Work backwards through the list
        timesteps = scheduler.timesteps  # [981, 961, ..., 21, 1]
        
        latents = []
        z0 = encode_image(self.setup.vae, image, self.setup.device, self.setup.dtype)
        latents.append(z0)
        
        text_embeddings = encode_prompt(
            self.setup.text_encoder,
            self.setup.tokenizer,
            source_prompt,
            self.setup.device,
        )
        
        zt = z0
        # ✅ Go backwards: from end to start
        for i in range(len(timesteps) - 1, 0, -1):
            t_curr = timesteps[i]      # 1, 21, 41, ...
            t_prev = timesteps[i - 1]  # 21, 41, 61, ... (this is actually "next" in inversion)
            
            noise_pred = predict_noise(
                self.setup.unet,
                zt,
                t_curr,
                text_embeddings,
                guidance_scale=self.config.inversion_guidance_scale,
            )
            zt = ddim_step_reverse(scheduler, noise_pred, t_curr, t_prev, zt)
            latents.append(zt)
        
        return {
            "latents": latents,
            "text_embeddings": text_embeddings,
            "uncond_embeddings": get_null_embedding(
                self.setup.text_encoder,
                self.setup.tokenizer,
                self.setup.device,
            ),
        }

    @torch.no_grad()
    def edit(
        self,
        inversion_result: Dict[str, Any],
        target_prompt: str,
        attention_controller: Optional[PromptToPromptController] = None,
    ) -> torch.Tensor:
        scheduler = self.setup.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        total_steps = len(scheduler.timesteps)
        if attention_controller is not None:
            attention_controller.set_total_steps(total_steps)

        zt = inversion_result["latents"][-1]
        target_embeddings = encode_prompt(
            self.setup.text_encoder,
            self.setup.tokenizer,
            target_prompt,
            self.setup.device,
        )
        uncond_embeddings = inversion_result["uncond_embeddings"]

        for idx, timestep in enumerate(scheduler.timesteps):
            if attention_controller is not None:
                attention_controller.set_step(idx)
            noise_pred = predict_noise(
                self.setup.unet,
                zt,
                timestep,
                target_embeddings,
                guidance_scale=self.config.guidance_scale,
                uncond_embeddings=uncond_embeddings,
            )
            zt = ddim_step_forward(scheduler, noise_pred, timestep, zt)

        decoded = decode_latent(self.setup.vae, zt, self.setup.device)
        return decoded


@register_model("ddim")
class DDIMEditor:
    """
    High-level interface consumed by run/test scripts.
    """

    def __init__(
        self,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        precision: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
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

        self.setup = StandardImageEditingSetup(model_id=model_id, device=resolved_device, dtype=dtype)
        self.inversion = DDIMInversion(self.setup, self.config)
        self._last_attention_stats: Optional[Dict[str, Any]] = None

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
        return torch.float16

    @staticmethod
    def _resolve_attention_dtype(name: str) -> torch.dtype:
        mapping = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(name.lower(), torch.float16)

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
        Perform DDIM inversion/editing on a single image path.
        """
        image = load_image(image_path, self.config.image_size)
        inversion_result = self.inversion.invert(image, prompt_src)
        controller_stats = None
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

        reconstructed_tensor = decode_latent(
            self.setup.vae, inversion_result["latents"][0], self.setup.device
        )

        source_pil = image
        reconstructed_pil = tensor_to_pil(reconstructed_tensor)
        edited_pil = tensor_to_pil(edited_tensor)

        if output_path:
            edited_pil.save(output_path)

        if verbose:
            print("DDIM edit complete:")
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
        controller = self._build_prompt_to_prompt_controller(prompt_tar, blend_word)
        if controller is None:
            edited_tensor = self.inversion.edit(inversion_result, prompt_tar)
            return edited_tensor, None

        with self._use_attention_controller(controller):
            controller.set_mode("record")
            self.inversion.edit(
                inversion_result,
                prompt_src,
                attention_controller=controller,
            )
            controller.set_mode("apply")
            edited_tensor = self.inversion.edit(
                inversion_result,
                prompt_tar,
                attention_controller=controller,
            )
        return edited_tensor, controller.summary()

    def get_last_attention_stats(self) -> Optional[Dict[str, Any]]:
        return self._last_attention_stats

    def _build_prompt_to_prompt_controller(
        self,
        prompt_tar: str,
        blend_word: Optional[str],
    ) -> Optional[PromptToPromptController]:
        if not self.config.use_prompt_to_prompt:
            return None
        controller = PromptToPromptController(
            self_replace_steps=self.config.self_replace_steps,
            cross_replace_steps=self.config.cross_replace_steps,
            store_self_attention=self.config.p2p_store_self_attention,
            attention_dtype=self._resolve_attention_dtype(self.config.p2p_attention_dtype),
            step_stride=max(1, self.config.p2p_attention_step_stride),
            layer_keywords=self.config.p2p_layer_keywords or None,
        )
        if blend_word:
            mask = self._compute_replace_mask(prompt_tar, blend_word)
            if mask is not None:
                controller.set_replace_token_mask(mask)
        return controller

    def _compute_replace_mask(self, prompt: str, blend_word: str) -> Optional[torch.Tensor]:
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

