#!/usr/bin/env python3
"""
Simple smoke-test for the standardized DDIM baseline on a single PIE-Bench image.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ddim import DDIMEditor


def parse_args():
    parser = argparse.ArgumentParser(description="Test DDIM baseline on one PIE-Bench sample.")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of DDIM steps (use fewer for CPU)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale during editing')
    parser.add_argument('--model_id', type=str, default=None,
                        help='Optional Hugging Face model id (defaults to SD v1.5)')
    return parser.parse_args()


def resolve_device(arg_device: str) -> str:
    if arg_device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return arg_device


def load_sample():
    data_path = Path("data/PIE-Bench_v1")
    annotation_file = data_path / "mapping_file.json"
    target_image_path = "0_random_140/000000000004.jpg"

    with annotation_file.open("r") as f:
        annotations = json.load(f)

    for img_id, item in annotations.items():
        if item.get('image_path') == target_image_path:
            return img_id, item
    raise ValueError(f"Could not find sample with image path {target_image_path}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"DDIM steps: {args.num_steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    sample_id, sample_item = load_sample()
    data_path = Path("data/PIE-Bench_v1")
    image_path = data_path / "annotation_images" / sample_item['image_path']
    prompt_src = sample_item['original_prompt'].replace('[', '').replace(']', '')
    prompt_tar = sample_item['editing_prompt'].replace('[', '').replace(']', '')
    blend_word = sample_item.get('blended_word', None)

    print(f"\n{'='*60}")
    print("Testing DDIM Baseline")
    print(f"{'='*60}")
    print(f"Sample ID: {sample_id}")
    print(f"Image: {image_path}")
    print(f"Source prompt: {prompt_src}")
    print(f"Target prompt: {prompt_tar}")
    print(f"Blend word: {blend_word}")
    print(f"{'='*60}\n")

    editor = DDIMEditor(
        device=device,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        model_id=args.model_id or "runwayml/stable-diffusion-v1-5",
    )

    print("Running DDIM edit...")
    result = editor.edit_image(
        image_path=str(image_path),
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        blend_word=blend_word,
        output_path=f"outputs/test_ddim_{sample_id}.png",
        return_intermediate=True,
        verbose=True,
    )

    if isinstance(result, dict):
        intermediate_dir = Path(f"outputs/intermediate_{sample_id}_ddim")
        intermediate_dir.mkdir(exist_ok=True, parents=True)
        for key, img in result.items():
            if isinstance(img, Image.Image):
                img.save(intermediate_dir / f"{key}.png")
                print(f"Saved intermediate image: {intermediate_dir / f'{key}.png'}")

    print(f"\n{'='*60}")
    print("DDIM test completed!")
    print(f"Output saved to: outputs/test_ddim_{sample_id}.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

