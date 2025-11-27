#!/usr/bin/env python3
"""
Run a registered image-editing model (DDIM by default) on the full PIE-Bench dataset.
"""

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

# Add parent directory to path (since we're in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import available_models, get_model_builder
from src.utils.logging_utils import log_error, setup_logger

def _build_editor(device: str, args: argparse.Namespace):
    model_names = available_models()
    if args.model not in model_names:
        raise ValueError(f"Unknown model '{args.model}'. Available models: {model_names}")
    ModelClass = get_model_builder(args.model)

    model_kwargs: Dict[str, object] = {
        "device": device,
        "num_inference_steps": args.num_steps,
    }
    if args.num_inversion_steps is not None:
        model_kwargs["num_inversion_steps"] = args.num_inversion_steps
    if args.model_id:
        model_kwargs["model_id"] = args.model_id
    if args.guidance_scale:
        model_kwargs["guidance_scale"] = args.guidance_scale
    if args.precision:
        model_kwargs["precision"] = args.precision
    
    # Handle config overrides including FreeU and Rescale
    config_overrides = {}
    if args.config_overrides:
        try:
            config_overrides = json.loads(args.config_overrides)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --config_overrides: {e}")
    
    # CLI arguments take precedence or merge into overrides
    if args.use_freeu:
        config_overrides["use_freeu"] = True
        config_overrides["freeu_b1"] = args.freeu_b1
        config_overrides["freeu_b2"] = args.freeu_b2
        config_overrides["freeu_s1"] = args.freeu_s1
        config_overrides["freeu_s2"] = args.freeu_s2
    
    if args.rescale_factor > 0:
        config_overrides["rescale_factor"] = args.rescale_factor

    if args.use_masactrl:
        config_overrides["use_masactrl"] = True
        config_overrides["masactrl_step_start"] = args.masactrl_step_start
        if args.masactrl_layer_keywords:
            config_overrides["masactrl_layer_keywords"] = args.masactrl_layer_keywords

    if args.use_latent_blending or args.latent_blend_steps > 0:
        config_overrides["use_latent_blending"] = True
        config_overrides["latent_blend_steps"] = args.latent_blend_steps

    if config_overrides:
        model_kwargs["config_overrides"] = config_overrides

    return ModelClass(**model_kwargs)


def main():
    model_choices = available_models()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ddim', choices=model_choices, help='Model to run')
    parser.add_argument('--model_id', type=str, default=None, help='Optional Hugging Face model id')
    parser.add_argument('--categories', nargs='+', type=str, default=['0','1','2','3','4','5','6','7','8','9'],
                       help='Categories to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=50, help='Number of editing steps (use 10-20 for CPU, 50 for CUDA)')
    parser.add_argument('--num_inversion_steps', type=int, default=None, help='Optional override for inversion steps (defaults to editing steps when unset)')
    parser.add_argument('--guidance_scale', type=float, default=None, help='Optional override for guidance scale')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default=None, help='Optional precision override')
    
    # FreeU args
    parser.add_argument('--use_freeu', action='store_true', help='Enable FreeU')
    parser.add_argument('--freeu_b1', type=float, default=1.2, help='FreeU b1')
    parser.add_argument('--freeu_b2', type=float, default=1.4, help='FreeU b2')
    parser.add_argument('--freeu_s1', type=float, default=0.9, help='FreeU s1')
    parser.add_argument('--freeu_s2', type=float, default=0.2, help='FreeU s2')
    
    # Rescale CFG
    parser.add_argument('--rescale_factor', type=float, default=0.0, help='Rescale CFG factor (0.0 = disabled, typically 0.7)')

    # MasaCtrl
    parser.add_argument('--use_masactrl', action='store_true', help='Enable MasaCtrl')
    parser.add_argument('--masactrl_step_start', type=int, default=0, help='Step to start MasaCtrl')
    parser.add_argument('--masactrl_layer_keywords', nargs='+', type=str, default=[], help='Layer keywords for MasaCtrl')

    # Latent Blending
    parser.add_argument('--use_latent_blending', action='store_true', help='Enable Latent Blending')
    parser.add_argument('--latent_blend_steps', type=int, default=15, help='Number of steps for Latent Blending')
    
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs for each image')
    parser.add_argument('--config_overrides', type=str, default=None, help='JSON string for model-specific config overrides (e.g., \'{"null_inner_steps": 20, "null_lr": 0.01}\')')
    args = parser.parse_args()
    
    # Detect device (skip MPS by default due to memory issues)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Model: {args.model}")
    inv_steps = args.num_inversion_steps or args.num_steps
    print(f"Device: {device}, Edit steps: {args.num_steps}, Inversion steps: {inv_steps}")
    print(f"Categories: {args.categories}")

    # Set up logger
    logger = setup_logger(log_file="logs/pie_bench.log")

    # Load annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by categories
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] in args.categories]
    
    print(f"\nProcessing {len(samples)} images total")
    
    # Initialize editor
    editor = _build_editor(device, args)
    
    # Process images
    success = 0
    errors = 0
    total = len(samples)
    progress = tqdm(samples, desc="Editing", unit="img", position=0, leave=True)
    start_time = time.perf_counter()
    for idx, (img_id, item) in enumerate(progress, start=1):
        progress.set_description(f"[{idx}/{total}] {img_id}")
        image_path = f"data/PIE-Bench_v1/annotation_images/{item['image_path']}"
        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        output_path = Path("outputs") / args.model / "annotation_images" / item['image_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
            log_error(logger, f"Error on {img_id}: {e}", exc_info=True)
            errors += 1
            continue

    progress.close()
    elapsed = time.perf_counter() - start_time
    
    print(f"\n{'='*60}")
    print(f"[DONE] Complete!")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Output: outputs/{args.model}/")
    elapsed_td = timedelta(seconds=elapsed)
    print(f"  Time elapsed: {elapsed_td}")
    if success:
        avg_per_image = elapsed / success
        print(f"  Avg per processed image: {avg_per_image:.2f} s")
    print(f"{'='*60}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

