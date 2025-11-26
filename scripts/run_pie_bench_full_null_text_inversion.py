#!/usr/bin/env python3
"""
Run a registered image-editing model on the full PIE-Bench dataset.
Supports DDIM, Null-Text, and others via the registry.
"""

import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import available_models, get_model_builder

Path("logs").mkdir(exist_ok=True)

# Setup simple logger
logging.basicConfig(
    filename="logs/pie_bench_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)

def _build_editor(device: str, args: argparse.Namespace):
    model_names = available_models()
    if args.model not in model_names:
        raise ValueError(f"Unknown model '{args.model}'. Available models: {model_names}")
    ModelClass = get_model_builder(args.model)

    model_kwargs: Dict[str, object] = {
        "device": device,
        "num_inference_steps": args.num_steps,
    }
    
    # Common arguments
    if args.num_inversion_steps is not None:
        model_kwargs["num_inversion_steps"] = args.num_inversion_steps
    if args.model_id:
        model_kwargs["model_id"] = args.model_id
    if args.guidance_scale:
        model_kwargs["guidance_scale"] = args.guidance_scale
    if args.precision:
        model_kwargs["precision"] = args.precision
    
    # === Null-Text Specific Arguments (Auto-injected if model is null_text) ===
    if args.model == "null_text":
        model_kwargs["null_inner_steps"] = args.null_inner_steps
        model_kwargs["null_lr"] = args.null_lr

    # Parse config_overrides if provided (for advanced usage)
    if args.config_overrides:
        try:
            overrides = json.loads(args.config_overrides)
            model_kwargs.update(overrides) # Update kwargs directly
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --config_overrides: {e}")

    return ModelClass(**model_kwargs)


def main():
    model_choices = available_models()
    parser = argparse.ArgumentParser(description="Run full PIE-Bench evaluation")
    
    # Core Model Args
    parser.add_argument('--model', type=str, default='ddim', choices=model_choices, help='Model to run')
    parser.add_argument('--model_id', type=str, default=None, help='Hugging Face model id')
    parser.add_argument('--categories', nargs='+', type=str, default=['0','1','2','3','4','5','6','7','8','9'],
                        help='List of category IDs to process (0-9)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    
    # Hyperparameters
    parser.add_argument('--num_steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--num_inversion_steps', type=int, default=None, help='Inversion steps (defaults to num_steps)')
    parser.add_argument('--guidance_scale', type=float, default=None, help='CFG scale')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp32', help='Precision (use fp32 for Null-Text)')
    
    # Null-Text Specifics
    parser.add_argument('--null_inner_steps', type=int, default=10, help='[Null-Text] Optimization steps per timestep')
    parser.add_argument('--null_lr', type=float, default=0.01, help='[Null-Text] Learning rate')

    # Misc
    parser.add_argument('--verbose', action='store_true', help='Print details to console')
    parser.add_argument('--resume', action='store_true', help='Skip images that already exist in output folder')
    parser.add_argument('--config_overrides', type=str, default=None, help='JSON string for extra config')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"{'='*40}")
    print(f"Model:     {args.model}")
    print(f"Device:    {device}")
    print(f"Precision: {args.precision}")
    print(f"Steps:     {args.num_steps}")
    if args.model == "null_text":
        print(f"Null-Text: {args.null_inner_steps} steps, lr={args.null_lr}")
    print(f"{'='*40}")

    # Ensure log directory
    Path("logs").mkdir(exist_ok=True)

    # Load Annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    if not Path(annotation_file).exists():
        print(f"Error: {annotation_file} not found.")
        sys.exit(1)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter Samples
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] in args.categories]
    print(f"\nFound {len(samples)} images for categories {args.categories}")
    
    # Build Editor
    try:
        editor = _build_editor(device, args)
    except Exception as e:
        print(f"Failed to initialize editor: {e}")
        sys.exit(1)
    
    # Run Loop
    success = 0
    errors = 0
    skipped = 0
    
    progress = tqdm(samples, desc="Running Benchmark", unit="img")
    start_time = time.perf_counter()
    
    for idx, (img_id, item) in enumerate(progress):
        progress.set_description(f"[{success+errors+skipped}/{len(samples)}] {img_id}")
        
        image_rel_path = item['image_path']
        image_path = f"data/PIE-Bench_v1/annotation_images/{image_rel_path}"
        
        # Output: outputs/null_text/annotation_images/...
        output_path = Path("outputs") / args.model / "annotation_images" / image_rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resume Logic
        if args.resume and output_path.exists():
            skipped += 1
            continue

        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        try:
            editor.edit_image(
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                blend_word=blend_word,
                output_path=str(output_path),
                verbose=args.verbose
            )
            success += 1
        except Exception as e:
            errors += 1
            err_msg = f"Error on {img_id}: {str(e)}"
            progress.write(err_msg)
            logger.error(err_msg)
            # Optional: Save a black image or placeholder? 
            # Usually better to fail silently so we know it's missing.
            continue

    progress.close()
    elapsed = time.perf_counter() - start_time
    
    print(f"\n{'='*60}")
    print(f"Benchmark Complete!")
    print(f"  Total:   {len(samples)}")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Output:  outputs/{args.model}/")
    print(f"  Time:    {timedelta(seconds=elapsed)}")
    if success > 0:
        print(f"  Avg/Img: {elapsed/success:.2f}s")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()