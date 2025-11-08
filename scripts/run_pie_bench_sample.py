#!/usr/bin/env python3
"""
Run Direct Inversion on a small sample of PIE-Bench (e.g., first 5-10 images).
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
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--category', type=str, default='0', help='Category to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=10, help='Number of DDIM steps (use 10 for CPU, 50 for CUDA)')
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
    
    print(f"Device: {device}, Steps: {args.num_steps}, Images: {args.num_images}")
    
    # Load annotations
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by category
    samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] == args.category]
    samples = samples[:args.num_images]
    
    print(f"\nProcessing {len(samples)} images from category {args.category}")
    
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
    
    # Process images
    total = len(samples)
    start_time = time.perf_counter()
    processed = 0
    errors = 0
    total_batches = (total + batch_size - 1) // batch_size
    
    # Main progress bar
    main_progress = tqdm(total=total, desc="Processing images", unit="img", position=0)
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = samples[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        
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
        
        # Process batch with progress callback
        try:
            # Create progress callback
            def progress_callback(current_in_batch, total_in_batch, img_id):
                # Update description and progress (one image done)
                main_progress.set_description(f"Batch {batch_num}/{total_batches} | {img_id[:40]}")
                main_progress.update(1)
            
            results = editor.edit_images_batch(
                image_paths=image_paths,
                prompt_srcs=prompt_srcs,
                prompt_tars=prompt_tars,
                blend_words=blend_words,
                output_paths=output_paths,
                verbose=args.verbose,
                progress_callback=progress_callback
            )
            
            # Count successes and errors (progress already updated by callback)
            for i, result in enumerate(results):
                if result is not None:
                    processed += 1
                else:
                    errors += 1
                    main_progress.write(f"Error processing {img_ids[i]}")
                    
        except Exception as e:
            main_progress.write(f"Error processing batch {batch_num}: {e}")
            errors += len(batch_samples)
            main_progress.update(len(batch_samples))  # Update progress even on error
            continue
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    main_progress.close()

    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Complete! Results in outputs/direct_inversion/")
    if processed:
        elapsed_td = timedelta(seconds=elapsed)
        avg_per_image = elapsed / processed
        total_images_full = len(annotations)
        estimated_total = avg_per_image * total_images_full
        print(f"  • Processed: {processed}/{total} images")
        if errors > 0:
            print(f"  • Errors: {errors}")
        print(f"  • Time for {processed} images: {elapsed_td}")
        print(f"  • Avg per image: {avg_per_image:.2f} s")
        print(f"  • Estimated full PIE-Bench ({total_images_full} imgs): {timedelta(seconds=estimated_total)}")
        print(f"    (~{estimated_total/3600:.1f} hours)")
    else:
        print("  • No images processed successfully; cannot estimate total time.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

