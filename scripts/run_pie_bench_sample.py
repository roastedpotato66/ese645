#!/usr/bin/env python3
"""
Run Direct Inversion on a small sample of PIE-Bench (e.g., first 5-10 images).
"""

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
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--category', type=str, default='0', help='Category to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_steps', type=int, default=10, help='Number of DDIM steps (use 10 for CPU, 50 for CUDA)')
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
    
    # Initialize editor
    editor = DirectInversionEditor(device=device, num_ddim_steps=args.num_steps)
    
    # Process images
    for img_id, item in tqdm(samples, desc="Editing"):
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
                output_path=output_path
            )
        except Exception as e:
            print(f"\nError on {img_id}: {e}")
            continue
    
    print(f"\n✓ Complete! Results in outputs/direct_inversion/")

if __name__ == '__main__':
    main()

