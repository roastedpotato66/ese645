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
from src.utils.device_utils import get_batch_size_with_override, print_batch_size_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', type=str, default=['0','1','2','3','4','5','6','7','8','9'],
                       help='Categories to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=50, help='Number of DDIM steps (use 10-20 for CPU, 50 for CUDA)')
    parser.add_argument('--batch_size', type=str, default='auto', 
                       help='Batch size for processing (auto=detect automatically, or integer like 4)')
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
    
    # Determine batch size
    batch_size = get_batch_size_with_override(
        device=device,
        num_ddim_steps=args.num_steps,
        batch_size_override=args.batch_size
    )
    
    # Print batch size info
    print_batch_size_info(device, args.num_steps, batch_size)
    
    # Initialize editor
    editor = DirectInversionEditor(device=device, num_ddim_steps=args.num_steps)
    
    # Process images in batches
    success = 0
    errors = 0
    total = len(samples)
    start_time = time.perf_counter()
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = samples[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (images {batch_start+1}-{batch_end} of {total})")
        
        # Prepare batch data
        image_paths = []
        prompt_srcs = []
        prompt_tars = []
        blend_words = []
        output_paths = []
        img_ids = []
        
        for img_id, item in batch_samples:
            image_paths.append(f"data/PIE-Bench_v1/annotation_images/{item['image_path']}")
            prompt_srcs.append(item['original_prompt'].replace('[', '').replace(']', ''))
            prompt_tars.append(item['editing_prompt'].replace('[', '').replace(']', ''))
            blend_words.append(item.get('blended_word', None))
            output_path = f"outputs/direct_inversion/annotation_images/{item['image_path']}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_paths.append(output_path)
            img_ids.append(img_id)
        
        # Process batch
        try:
            results = editor.edit_images_batch(
                image_paths=image_paths,
                prompt_srcs=prompt_srcs,
                prompt_tars=prompt_tars,
                blend_words=blend_words,
                output_paths=output_paths,
                verbose=args.verbose
            )
            
            # Count successes and errors
            for i, result in enumerate(results):
                if result is not None:
                    success += 1
                else:
                    errors += 1
                    print(f"Error processing {img_ids[i]}")
                    
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            errors += len(batch_samples)
            continue
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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

