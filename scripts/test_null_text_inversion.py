#!/usr/bin/env python3
"""
Simple smoke-test for the Null Text Inversion baseline on a single PIE-Bench image.
"""

import argparse
import json
import sys
from pathlib import Path
from contextlib import contextmanager

import torch
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Assumption: The class is named NullTextEditor and located in src/models/null_text.py
from src.models.null_text import NullTextEditor


def parse_args():
    parser = argparse.ArgumentParser(description="Test Null Text Inversion baseline on one PIE-Bench sample.")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of inference steps (standard is 50)')
    parser.add_argument('--num_inversion_steps', type=int, default=50,
                        help='Number of inversion steps (Null Text typically matches num_steps)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale during editing')
    # Null Text specific arguments
    parser.add_argument('--null_inner_steps', type=int, default=10,
                        help='Number of optimization steps for null-text embedding per timestep')
    parser.add_argument('--null_lr', type=float, default=1e-2,
                        help='Learning rate for null-text optimization')
    
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
    # Keep strictly the same sample as DDIM for fair comparison
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
    print(f"Inference steps: {args.num_steps}")
    print(f"Null Optimization steps: {args.null_inner_steps}")
    print(f"Null Learning Rate: {args.null_lr}")

    sample_id, sample_item = load_sample()
    data_path = Path("data/PIE-Bench_v1")
    image_path = data_path / "annotation_images" / sample_item['image_path']
    prompt_src = sample_item['original_prompt'].replace('[', '').replace(']', '')
    prompt_tar = sample_item['editing_prompt'].replace('[', '').replace(']', '')
    # Null Text usually relies on Prompt-to-Prompt logic, so blend_word is still useful if implemented
    blend_word = sample_item.get('blended_word', None)

    print(f"\n{'='*60}")
    print("Testing Null Text Inversion Baseline")
    print(f"{'='*60}")
    print(f"Sample ID: {sample_id}")
    print(f"Image: {image_path}")
    print(f"Source prompt: {prompt_src}")
    print(f"Target prompt: {prompt_tar}")
    print(f"Blend word: {blend_word}")
    print(f"{'='*60}\n")

    # Initialize the Null Text Editor
    # Note: Ensure your NullTextEditor class __init__ accepts these arguments
    editor = NullTextEditor(
        device=device,
        num_inference_steps=args.num_steps,
        num_inversion_steps=args.num_inversion_steps,
        guidance_scale=args.guidance_scale,
        model_id=args.model_id or "runwayml/stable-diffusion-v1-5",
        null_inner_steps=args.null_inner_steps,
        null_lr=args.null_lr
    )

    print("Running Null Text Inversion & Edit (this may take longer than DDIM)...")
    result = editor.edit_image(
        image_path=str(image_path),
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        blend_word=blend_word,
        output_path=f"outputs/test_null_text_{sample_id}.png",
        return_intermediate=True,
        verbose=True,
    )

    if isinstance(result, dict):
        intermediate_dir = Path(f"outputs/intermediate_{sample_id}_null_text")
        intermediate_dir.mkdir(exist_ok=True, parents=True)
        for key, img in result.items():
            if isinstance(img, Image.Image):
                img.save(intermediate_dir / f"{key}.png")
                print(f"Saved intermediate image: {intermediate_dir / f'{key}.png'}")

    print(f"\n{'='*60}")
    print("Null Text test completed!")
    print(f"Output saved to: outputs/test_null_text_{sample_id}.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()