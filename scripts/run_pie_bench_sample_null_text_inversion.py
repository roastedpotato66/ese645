#!/usr/bin/env python3
"""
Run Null-Text Inversion on a sample of PIE-Bench.
"""

import time
from datetime import timedelta

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

# Add parent directory to path (since we're in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import available_models, get_model_builder

def _build_editor(device: str, args: argparse.Namespace):
    model_names = available_models()
    if args.model not in model_names:
        raise ValueError(f"Unknown model '{args.model}'. Available models: {model_names}")
    
    ModelClass = get_model_builder(args.model)

    model_kwargs: Dict[str, object] = {
        "device": device,
        "num_inference_steps": args.num_steps,
    }
    
    # Optional base arguments
    if args.num_inversion_steps is not None:
        model_kwargs["num_inversion_steps"] = args.num_inversion_steps
    if args.model_id:
        model_kwargs["model_id"] = args.model_id
    if args.guidance_scale:
        model_kwargs["guidance_scale"] = args.guidance_scale
    if args.precision:
        model_kwargs["precision"] = args.precision

    # === Null-Text Specific Arguments ===
    if args.model == 'null_text':
        # These arguments must match the __init__ in src/models/null_text.py
        model_kwargs["null_inner_steps"] = args.null_inner_steps
        model_kwargs["null_lr"] = args.null_lr

    return ModelClass(**model_kwargs)


def main():
    model_choices = available_models()
    parser = argparse.ArgumentParser(description="Run Null-Text Inversion on PIE-Bench")
    
    # Standard arguments
    parser.add_argument('--model', type=str, default='null_text', choices=model_choices, help='Model to run')
    parser.add_argument('--model_id', type=str, default=None, help='Optional Hugging Face model id')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--category', type=str, default='0', help='Category to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=50, help='Number of editing/inversion steps (Standard for Null-Text is 50)')
    parser.add_argument('--num_inversion_steps', type=int, default=None, help='Optional override for inversion steps (defaults to num_steps)')
    parser.add_argument('--guidance_scale', type=float, default=None, help='Optional override for guidance scale')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp32', help='Precision (recommend fp32 to avoid black images)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs for each image')
    
    # Null-Text specific arguments
    parser.add_argument('--null_inner_steps', type=int, default=10, help='Number of optimization steps per timestep')
    parser.add_argument('--null_lr', type=float, default=0.01, help='Learning rate for null optimization')

    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Model: {args.model}")
    inv_steps = args.num_inversion_steps or args.num_steps
    print(f"Device: {device}")
    print(f"Config: Steps={args.num_steps}, InvSteps={inv_steps}, NullSteps={args.null_inner_steps}, LR={args.null_lr}")
    
    # Load annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    if not Path(annotation_file).exists():
        print(f"Error: Annotation file not found at {annotation_file}")
        sys.exit(1)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by category
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] == args.category]
    samples = samples[:args.num_images]
    
    print(f"\nProcessing {len(samples)} images from category {args.category}")
    
    # Initialize editor
    try:
        editor = _build_editor(device, args)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Process images
    total = len(samples)
    progress = tqdm(samples, desc="Editing", unit="img")
    start_time = time.perf_counter()
    processed = 0
    
    for idx, (img_id, item) in enumerate(progress, start=1):
        progress.set_description(f"[{idx}/{total}] {img_id}")
        image_path = f"data/PIE-Bench_v1/annotation_images/{item['image_path']}"
        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        # Output structure: outputs/null_text/annotation_images/...
        output_path = Path("outputs") / args.model / "annotation_images" / item['image_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists (optional, handy for resuming)
        # if output_path.exists():
        #     continue

        try:
            editor.edit_image(
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                blend_word=blend_word,
                output_path=str(output_path),
                verbose=args.verbose
            )
            processed += 1
        except Exception as e:
            progress.write(f"Error on {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    progress.close()

    elapsed = time.perf_counter() - start_time
    print(f"\n✓ Complete! Results in outputs/{args.model}/")
    if processed:
        elapsed_td = timedelta(seconds=elapsed)
        avg_per_image = elapsed / processed
        total_images_full = len(annotations)
        estimated_total = avg_per_image * total_images_full
        print(f"  • Time for {processed} images: {elapsed_td}")
        print(f"  • Avg per image: {avg_per_image:.2f} s")
        print(f"  • Estimated full PIE-Bench ({total_images_full} imgs): {timedelta(seconds=estimated_total)}")
    else:
        print("  • No images processed successfully.")

if __name__ == '__main__':
    main()