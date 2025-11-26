#!/usr/bin/env python3
"""
Test script for Direct Inversion.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import get_model_builder


def parse_args():
    parser = argparse.ArgumentParser(description="Test Direct Inversion on one PIE-Bench sample.")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of Direct Inversion editing steps (use fewer for CPU)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale during editing')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to source image (defaults to PIE-Bench sample)')
    parser.add_argument('--prompt_src', type=str, default=None,
                        help='Source prompt (defaults to PIE-Bench sample)')
    parser.add_argument('--prompt_tar', type=str, default=None,
                        help='Target prompt (defaults to PIE-Bench sample)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path (defaults to outputs/test_direct_inversion_{sample_id}.png)')
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

    # Load defaults from PIE-Bench if not provided
    if args.image_path is None or args.prompt_src is None or args.prompt_tar is None:
        sample_id, sample_item = load_sample()
        data_path = Path("data/PIE-Bench_v1")
        
        if args.image_path is None:
            args.image_path = str(data_path / "annotation_images" / sample_item['image_path'])
        if args.prompt_src is None:
            args.prompt_src = sample_item['original_prompt'].replace('[', '').replace(']', '')
        if args.prompt_tar is None:
            args.prompt_tar = sample_item['editing_prompt'].replace('[', '').replace(']', '')
        if args.output_path is None:
            args.output_path = f"outputs/test_direct_inversion_{sample_id}.png"
        
        blend_word = sample_item.get('blended_word', None)
    else:
        sample_id = None
        blend_word = None
        if args.output_path is None:
            args.output_path = "outputs/test_direct_inversion.jpg"

    print(f"Using device: {device}")
    print(f"Editing steps: {args.num_steps}")
    print(f"Inversion steps: {args.num_steps}")  # Inversion steps should match inference for direct path mapping
    print(f"Guidance scale: {args.guidance_scale}")

    if sample_id is not None:
        print(f"\n{'='*60}")
        print("Testing Direct Inversion")
        print(f"{'='*60}")
        print(f"Sample ID: {sample_id}")
        print(f"Image: {args.image_path}")
        print(f"Source prompt: {args.prompt_src}")
        print(f"Target prompt: {args.prompt_tar}")
        print(f"Blend word: {blend_word}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("Testing Direct Inversion")
        print(f"{'='*60}")
        print(f"Image: {args.image_path}")
        print(f"Source prompt: {args.prompt_src}")
        print(f"Target prompt: {args.prompt_tar}")
        print(f"{'='*60}\n")

    print("Initializing Direct Inversion model...")
    
    # Build Direct Inversion model
    model_cls = get_model_builder("direct_inversion")
    editor = model_cls(
        device=device,
        num_inference_steps=args.num_steps,
        num_inversion_steps=args.num_steps,  # Inversion steps should match inference for direct path mapping
        guidance_scale=args.guidance_scale,
        model_id=args.model_id or "runwayml/stable-diffusion-v1-5",
    )

    print("Running Direct Inversion edit...")
    result = editor.edit_image(
        image_path=args.image_path,
        prompt_src=args.prompt_src,
        prompt_tar=args.prompt_tar,
        blend_word=blend_word,
        output_path=args.output_path,
        verbose=True
    )

    if isinstance(result, dict):
        if sample_id is not None:
            intermediate_dir = Path(f"outputs/intermediate_{sample_id}_direct_inversion")
        else:
            intermediate_dir = Path("outputs/intermediate_direct_inversion")
        intermediate_dir.mkdir(exist_ok=True, parents=True)
        for key, img in result.items():
            if isinstance(img, Image.Image):
                img.save(intermediate_dir / f"{key}.png")
                print(f"Saved intermediate image: {intermediate_dir / f'{key}.png'}")

    print(f"\n{'='*60}")
    print("Direct Inversion test completed!")
    print(f"Output saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()