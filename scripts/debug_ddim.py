#!/usr/bin/env python3
"""
Comprehensive debugging utility for the standardized DDIM baseline.

This script mirrors the stages outlined in `ddim_debug.md` and produces:
- Detailed logs (JSON) describing each stage of DDIM inversion/editing
- Intermediate plots/figures for latent trajectories and CFG sweeps
- Reconstructed / edited images for visual inspection
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ddim import DDIMEditor
from src.utils.ddim_utils import (
    decode_latent,
    encode_image,
    encode_prompt,
    get_null_embedding,
    load_image,
    predict_noise,
    tensor_to_pil,
)


def _resolve_sample(args) -> Tuple[str, Dict]:
    annotation_file = Path("data/PIE-Bench_v1/mapping_file.json")
    with annotation_file.open("r") as f:
        annotations = json.load(f)

    if args.sample_id and args.sample_id in annotations:
        return args.sample_id, annotations[args.sample_id]

    if args.image_path:
        for sample_id, item in annotations.items():
            if item.get("image_path") == args.image_path:
                return sample_id, item

    default_path = "0_random_140/000000000004.jpg"
    for sample_id, item in annotations.items():
        if item.get("image_path") == default_path:
            return sample_id, item

    raise ValueError("Could not resolve sample. Please pass --sample-id or --image-path.")


class DDIMDebugRunner:
    def __init__(self, editor: DDIMEditor, output_dir: Path, blend_word: Optional[str]):
        self.editor = editor
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs: List[Dict] = []
        self.transforms = transforms.ToTensor()
        self.blend_word = blend_word

    def log(self, message: str, data: Optional[object] = None):
        entry = {"message": message}
        if data is not None:
            entry["data"] = data
        self.logs.append(entry)
        print(f"[DEBUG] {message}")
        if data is not None:
            print(f"        {data}")

    def save_log(self):
        with (self.output_dir / "debug_log.json").open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2)

    # ---- Stage implementations -------------------------------------------------
    def stage_input(self, image: Image.Image, source_prompt: str) -> torch.Tensor:
        self.log("Stage 1", "Input validation")
        self.log(
            "Prompt-to-Prompt config",
            {
                "enabled": self.editor.config.use_prompt_to_prompt,
                "self_replace_steps": self.editor.config.self_replace_steps,
                "cross_replace_steps": self.editor.config.cross_replace_steps,
                "blend_word": self.blend_word,
            },
        )
        self.log("Image type", str(type(image)))
        self.log("Image size", image.size)
        self.log("Image mode", image.mode)
        np_img = np.array(image)
        self.log("Image range", f"[{np_img.min()}, {np_img.max()}]")
        image.save(self.output_dir / "1_input_image.png")

        text_emb = encode_prompt(
            self.editor.setup.text_encoder,
            self.editor.setup.tokenizer,
            source_prompt,
            self.editor.setup.device,
        )
        self.log(
            "Text embedding stats",
            {
                "shape": list(text_emb.shape),
                "dtype": str(text_emb.dtype),
                "min": float(text_emb.min()),
                "max": float(text_emb.max()),
                "mean": float(text_emb.mean()),
                "std": float(text_emb.std()),
            },
        )

        tokens = self.editor.setup.tokenizer(
            source_prompt,
            padding="max_length",
            max_length=self.editor.setup.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.log("Token IDs (first 20)", tokens.input_ids[0].tolist()[:20])
        self.log("Decoded tokens", self.editor.setup.tokenizer.decode(tokens.input_ids[0]))
        return text_emb

    def stage_vae(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        self.log("Stage 2", "VAE encoding/decoding")
        z0 = encode_image(
            self.editor.setup.vae,
            image,
            self.editor.setup.device,
            self.editor.setup.dtype,
        )

        self.log(
            "Latent z0 stats",
            {
                "shape": list(z0.shape),
                "dtype": str(z0.dtype),
                "min": float(z0.min()),
                "max": float(z0.max()),
                "mean": float(z0.mean()),
                "std": float(z0.std()),
            },
        )

        scaling_factor = self.editor.setup.vae.config.scaling_factor
        self.log("VAE scaling factor", scaling_factor)
        if abs(scaling_factor - 0.18215) > 0.01:
            self.log("WARNING", "Unexpected scaling factor for SD VAE.")

        reconstructed = decode_latent(self.editor.setup.vae, z0, self.editor.setup.device)
        recon_img = tensor_to_pil(reconstructed)
        recon_img.save(self.output_dir / "2_vae_reconstruction.png")
        self.log(
            "Reconstructed tensor stats",
            {
                "shape": list(reconstructed.shape),
                "min": float(reconstructed.min()),
                "max": float(reconstructed.max()),
            },
        )

        original_tensor = self.transforms(image).unsqueeze(0).to(self.editor.setup.device)
        mse = F.mse_loss(reconstructed, original_tensor)
        self.log("VAE reconstruction MSE", float(mse))
        return z0, reconstructed

    def stage_noise_prediction(self, z0: torch.Tensor, text_emb: torch.Tensor):
        self.log("Stage 3", "Noise prediction sanity checks")
        scheduler = self.editor.setup.scheduler
        scheduler.set_timesteps(self.editor.config.num_inference_steps)
        timesteps_to_test = [0, 10, 25, 40, self.editor.config.num_inference_steps - 1]
        timesteps_to_test = [t for t in timesteps_to_test if t < len(scheduler.timesteps)]

        uncond_emb = get_null_embedding(
            self.editor.setup.text_encoder,
            self.editor.setup.tokenizer,
            self.editor.setup.device,
        )

        for idx in timesteps_to_test:
            timestep = scheduler.timesteps[idx]
            noise_no_cfg = predict_noise(
                self.editor.setup.unet,
                z0,
                timestep,
                text_emb,
                guidance_scale=1.0,
            )
            noise_cfg = predict_noise(
                self.editor.setup.unet,
                z0,
                timestep,
                text_emb,
                guidance_scale=self.editor.config.guidance_scale,
                uncond_embeddings=uncond_emb,
            )
            amplification = (noise_cfg.abs().mean() / noise_no_cfg.abs().mean()).item()
            self.log(
                f"Noise stats at timestep {int(timestep)}",
                {
                    "no_cfg_range": [float(noise_no_cfg.min()), float(noise_no_cfg.max())],
                    "cfg_range": [float(noise_cfg.min()), float(noise_cfg.max())],
                    "cfg_amplification": amplification,
                },
            )

    def stage_inversion(self, image: Image.Image, source_prompt: str) -> Dict[str, object]:
        self.log("Stage 4", "DDIM inversion trajectory")
        inversion_result = self.editor.inversion.invert(image, source_prompt)
        latents: List[torch.Tensor] = inversion_result["latents"]
        self.log("Latent steps", len(latents))

        stats = []
        for idx, latent in enumerate(latents):
            latent = latent.detach()
            stat = {
                "step": idx,
                "min": float(latent.min()),
                "max": float(latent.max()),
                "mean": float(latent.mean()),
                "std": float(latent.std()),
            }
            stats.append(stat)
            if idx % 10 == 0 or idx == len(latents) - 1:
                self.log(f"Latent stats @ step {idx}", stat)

        # Plot latent ranges/mean/std
        steps = [s["step"] for s in stats]
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(steps, [s["min"] for s in stats], label="min")
        plt.plot(steps, [s["max"] for s in stats], label="max")
        plt.legend()
        plt.ylabel("Latent value")
        plt.title("Latent range during inversion")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(steps, [s["mean"] for s in stats])
        plt.ylabel("Mean")
        plt.title("Latent mean")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(steps, [s["std"] for s in stats])
        plt.xlabel("Step")
        plt.ylabel("Std")
        plt.title("Latent std")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "4_latent_stats.png", dpi=150)
        plt.close()

        # Histogram of final latent
        zT = latents[-1].detach().flatten().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(zT, bins=60, density=True, alpha=0.7, label="zT")
        x = np.linspace(zT.min(), zT.max(), 200)
        plt.plot(x, 1 / math.sqrt(2 * math.pi) * np.exp(-x**2 / 2), "r-", label="N(0,1)")
        plt.title("zT distribution vs Gaussian")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "4_zT_hist.png", dpi=150)
        plt.close()

        self.log(
            "zT stats",
            {"mean": float(zT.mean()), "std": float(zT.std()), "abs_mean": float(np.abs(zT.mean()))},
        )
        return inversion_result

    def stage_reconstruction(
        self,
        inversion_result: Dict[str, object],
        source_prompt: str,
        original_image: Image.Image,
    ):
        self.log("Stage 5", "Reconstruction with same prompt")
        reconstructed = self.editor.inversion.edit(inversion_result, source_prompt)
        recon_img = tensor_to_pil(reconstructed)
        recon_img.save(self.output_dir / "5_reconstruction_same_prompt.png")

        original_tensor = self.transforms(original_image).unsqueeze(0).to(self.editor.setup.device)
        mse = F.mse_loss(reconstructed, original_tensor)
        psnr = 10 * torch.log10(torch.tensor(1.0, device=self.editor.setup.device) / mse)
        self.log("Reconstruction metrics", {"mse": float(mse), "psnr": float(psnr)})

    def stage_editing(
        self,
        inversion_result: Dict[str, object],
        source_prompt: str,
        target_prompt: str,
    ):
        self.log("Stage 6", "Editing with target prompt")
        self.log("Source prompt", source_prompt)
        self.log("Target prompt", target_prompt)
        target_emb = encode_prompt(
            self.editor.setup.text_encoder,
            self.editor.setup.tokenizer,
            target_prompt,
            self.editor.setup.device,
        )
        emb_diff = torch.mean(torch.abs(target_emb - inversion_result["text_embeddings"]))
        self.log("Text embedding difference", float(emb_diff))

        original_guidance = self.editor.config.guidance_scale
        cfg_values = [1.0, 3.0, 5.0, original_guidance, 10.0]

        edited_default, p2p_stats = (
            self.editor.apply_prompt_to_prompt(
                inversion_result,
                source_prompt,
                target_prompt,
                blend_word=self.blend_word,
            )
            if self.editor.config.use_prompt_to_prompt
            else (self.editor.inversion.edit(inversion_result, target_prompt), None)
        )
        if p2p_stats:
            self.log("Prompt-to-Prompt stats (default guidance)", p2p_stats)

        fig, axes = plt.subplots(1, len(cfg_values), figsize=(4 * len(cfg_values), 4))
        for idx, guidance in enumerate(cfg_values):
            self.editor.config.guidance_scale = guidance
            if guidance == original_guidance:
                edited = edited_default
            else:
                if self.editor.config.use_prompt_to_prompt:
                    edited, _ = self.editor.apply_prompt_to_prompt(
                        inversion_result,
                        source_prompt,
                        target_prompt,
                        blend_word=self.blend_word,
                    )
                else:
                    edited = self.editor.inversion.edit(inversion_result, target_prompt)
            edited_pil = tensor_to_pil(edited)
            edited_pil.save(self.output_dir / f"6_edit_cfg_{guidance:.2f}.png")
            axes[idx].imshow(edited_pil)
            axes[idx].set_title(f"CFG={guidance:.2f}")
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "6_guidance_grid.png", dpi=150)
        plt.close()
        self.editor.config.guidance_scale = original_guidance

        # Default edit
        tensor_to_pil(edited_default).save(self.output_dir / "6_edit_default.png")
        self.log(
            "Edited tensor stats",
            {
                "min": float(edited_default.min()),
                "max": float(edited_default.max()),
                "mean": float(edited_default.mean()),
            },
        )

    def stage_attention(
        self,
        inversion_result: Dict[str, object],
        source_prompt: str,
        target_prompt: str,
    ):
        self.log("Stage 7", "Cross-attention analysis")
        attention_maps = {"source": [], "target": [], "current": []}

        def hook_fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            attention_maps["current"].append(output.detach().float().mean().item())

        hooks = []
        for name, module in self.editor.setup.unet.named_modules():
            if "attn2" in name:
                hooks.append(module.register_forward_hook(hook_fn))

        # Source attention
        attention_maps["current"] = []
        _ = self.editor.inversion.edit(inversion_result, source_prompt)
        attention_maps["source"] = attention_maps["current"]

        # Target attention
        attention_maps["current"] = []
        _ = self.editor.inversion.edit(inversion_result, target_prompt)
        attention_maps["target"] = attention_maps["current"]

        for hook in hooks:
            hook.remove()

        if attention_maps["source"] and attention_maps["target"]:
            diffs = []
            for src, tgt in zip(attention_maps["source"], attention_maps["target"]):
                diffs.append(abs(src - tgt))
            avg_diff = sum(diffs) / len(diffs)
            self.log("Average attention diff", avg_diff)
        else:
            self.log("Attention warning", "No attention data captured. Hook mismatch?")

    def run_all(self, image: Image.Image, source_prompt: str, target_prompt: str):
        text_emb = self.stage_input(image, source_prompt)
        z0, _ = self.stage_vae(image)
        self.stage_noise_prediction(z0, text_emb)
        inversion_result = self.stage_inversion(image, source_prompt)
        self.stage_reconstruction(inversion_result, source_prompt, image)
        self.stage_editing(inversion_result, source_prompt, target_prompt)
        self.stage_attention(inversion_result, source_prompt, target_prompt)
        self.save_log()
        print(f"\nDebug artifacts saved to: {self.output_dir}")


def build_editor(args) -> DDIMEditor:
    overrides: Dict[str, Any] = {}
    if args.disable_p2p:
        overrides["use_prompt_to_prompt"] = False
    return DDIMEditor(
        device=args.device,
        num_inference_steps=args.num_steps,
        num_inversion_steps=args.num_inversion_steps,
        guidance_scale=args.guidance_scale,
        model_id=args.model_id or "runwayml/stable-diffusion-v1-5",
        config_overrides=overrides,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="DDIM debugging tool")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num_steps", type=int, default=50, help="Number of editing steps")
    parser.add_argument("--num_inversion_steps", type=int, default=100, help="Number of inversion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None, help="PIE-Bench sample id")
    parser.add_argument("--image_path", type=str, default=None, help="PIE-Bench relative image path")
    parser.add_argument("--output_dir", type=str, default="debug_outputs/ddim")
    parser.add_argument("--disable_p2p", action="store_true", help="Disable Prompt-to-Prompt attention control")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_id, sample_item = _resolve_sample(args)
    print(f"Using sample {sample_id}: {sample_item['image_path']}")
    editor = build_editor(args)
    output_dir = Path(args.output_dir) / sample_id

    image_path = Path("data/PIE-Bench_v1/annotation_images") / sample_item["image_path"]
    image = load_image(str(image_path), image_size=editor.config.image_size)
    prompt_src = sample_item["original_prompt"].replace("[", "").replace("]", "")
    prompt_tar = sample_item["editing_prompt"].replace("[", "").replace("]", "")

    blend_word = sample_item.get("blended_word")
    runner = DDIMDebugRunner(editor, output_dir, blend_word)
    runner.run_all(image, prompt_src, prompt_tar)


if __name__ == "__main__":
    main()

