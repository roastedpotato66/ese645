#!/usr/bin/env python3
"""
Run Direct Inversion on full PIE-Bench dataset (all 700 images).
"""

import time
from datetime import timedelta

import torch
import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path (since we're in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.direct_inversion import DirectInversionEditor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', type=str, default=['0','1','2','3','4','5','6','7','8','9'],
                       help='Categories to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=50, help='Number of DDIM steps (use 10-20 for CPU, 50 for CUDA)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs for each image')
    args = parser.parse_args()
    
    # Detect device (skip MPS by default due to memory issues)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}, Steps: {args.num_steps}")
    print(f"Categories: {args.categories}")
    
    # Load annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by categories
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] in args.categories]
    
    print(f"\nProcessing {len(samples)} images total")
    
    # Initialize editor
    editor = DirectInversionEditor(device=device, num_ddim_steps=args.num_steps)
    
    # Process images
    success = 0
    errors = 0
    total = len(samples)
    progress = tqdm(samples, desc="Editing", unit="img")
    start_time = time.perf_counter()
    for idx, (img_id, item) in enumerate(progress, start=1):
        progress.set_description(f"[{idx}/{total}] {img_id}")
        image_path = f"data/PIE-Bench_v1/annotation_images/{item['image_path']}"
        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        output_path = f"outputs/direct_inversion/annotation_images/{item['image_path']}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            editor.edit_image(
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                blend_word=blend_word,
                output_path=output_path,
                verbose=args.verbose
            )
            success += 1
        except Exception as e:
            progress.write(f"Error on {img_id}: {e}")
            errors += 1
            continue

    progress.close()
    elapsed = time.perf_counter() - start_time
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Output: outputs/direct_inversion/")
    elapsed_td = timedelta(seconds=elapsed)
    print(f"  Time elapsed: {elapsed_td}")
    if success:
        avg_per_image = elapsed / success
        print(f"  Avg per processed image: {avg_per_image:.2f} s")
    print(f"{'='*60}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

