#!/usr/bin/env python3
"""
Run a registered image-editing model (DDIM by default) on a sample of PIE-Bench.
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
            # Handle case where user passes quoted string "word1 word2" instead of multiple args
            keywords = []
            for k in args.masactrl_layer_keywords:
                keywords.extend(k.split())
            config_overrides["masactrl_layer_keywords"] = keywords

    if args.use_latent_blending:
        config_overrides["use_latent_blending"] = True
        config_overrides["latent_blend_steps"] = args.latent_blend_steps

    # DEBUG: Print active configuration to verify parsing
    print("\n--- Configuration Debug ---")
    print(f"Model: {args.model}")
    print(f"Use FreeU: {config_overrides.get('use_freeu', False)}")
    print(f"Use MasaCtrl: {config_overrides.get('use_masactrl', False)}")
    if config_overrides.get('use_masactrl'):
        print(f"  - Start Step: {config_overrides.get('masactrl_step_start')}")
        print(f"  - Keywords: {config_overrides.get('masactrl_layer_keywords')}")
    print(f"Use Latent Blending: {config_overrides.get('use_latent_blending', False)}")
    if config_overrides.get('use_latent_blending'):
        print(f"  - Steps: {config_overrides.get('latent_blend_steps')}")
    print("---------------------------\n")

    if config_overrides:
        model_kwargs["config_overrides"] = config_overrides

    return ModelClass(**model_kwargs)


def main():
    model_choices = available_models()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ddim', choices=model_choices, help='Model to run')
    parser.add_argument('--model_id', type=str, default=None, help='Optional Hugging Face model id')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--category', type=str, default='0', help='Category to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=10, help='Number of editing steps (use 10 for CPU, 50 for CUDA)')
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
    parser.add_argument('--output_path', type=str, default=None, help='Base output path (default: outputs/{model}/annotation_images)')
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
    print(f"Device: {device}, Edit steps: {args.num_steps}, Inversion steps: {inv_steps}, Images: {args.num_images}")
    
    # Load annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by category
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] == args.category]
    samples = samples[:args.num_images]
    
    print(f"\nProcessing {len(samples)} images from category {args.category}")
    
    # Initialize editor
    editor = _build_editor(device, args)
    
    # Process images
    total = len(samples)
    progress = tqdm(samples, desc="Editing", unit="img", position=0, leave=True)
    start_time = time.perf_counter()
    processed = 0
    for idx, (img_id, item) in enumerate(progress, start=1):
        progress.set_description(f"[{idx}/{total}] {img_id}")
        image_path = f"data/PIE-Bench_v1/annotation_images/{item['image_path']}"
        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        if args.output_path:
            output_path = Path(args.output_path) / "annotation_images" / item['image_path']
        else:
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
            processed += 1
        except Exception as e:
            progress.write(f"Error on {img_id}: {e}")
            continue

    progress.close()

    elapsed = time.perf_counter() - start_time
    if args.output_path:
        print(f"\n[DONE] Complete! Results in {args.output_path}")
    else:
        print(f"\n[DONE] Complete! Results in outputs/{args.model}/")
    if processed:
        elapsed_td = timedelta(seconds=elapsed)
        avg_per_image = elapsed / processed
        total_images_full = len(annotations)
        estimated_total = avg_per_image * total_images_full
        print(f"  - Time for {processed} images: {elapsed_td}")
        print(f"  - Avg per image: {avg_per_image:.2f} s")
        print(f"  - Estimated full PIE-Bench ({total_images_full} imgs): {timedelta(seconds=estimated_total)}")
    else:
        print("  - No images processed successfully; cannot estimate total time.")

if __name__ == '__main__':
    main()

