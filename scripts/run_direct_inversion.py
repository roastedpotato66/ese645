#!/usr/bin/env python3
"""
Run Direct Inversion on PIE-Bench dataset.

Usage:
    python scripts/run_direct_inversion.py --edit_categories 0 --num_images 5
    
This script will:
1. Load images from PIE-Bench dataset
2. Run Direct Inversion + P2P editing
3. Save edited images to outputs/direct_inversion/
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.direct_inversion import DirectInversionEditor
from src.utils.image_utils import setup_seed


def main():
    parser = argparse.ArgumentParser(description='Run Direct Inversion on PIE-Bench')
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/PIE-Bench_v1',
        help='Path to PIE-Bench dataset'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='outputs/direct_inversion',
        help='Path to save edited images'
    )
    parser.add_argument(
        '--edit_categories',
        nargs='+',
        type=str,
        default=['0'],
        help='Editing categories to process (0-9)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help='Number of images to process (None = all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--num_ddim_steps',
        type=int,
        default=50,
        help='Number of DDIM steps'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
    
    # Load annotations
    annotation_file = os.path.join(args.data_path, 'mapping_file.json')
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter by categories
    filtered_annotations = {
        k: v for k, v in annotations.items()
        if v['editing_type_id'] in args.edit_categories
    }
    
    if args.num_images:
        filtered_annotations = dict(list(filtered_annotations.items())[:args.num_images])
    
    print(f"\nProcessing {len(filtered_annotations)} images")
    print(f"Categories: {args.edit_categories}")
    print(f"Device: {device}")
    print(f"DDIM Steps: {args.num_ddim_steps}")
    
    # Initialize editor
    print("\nInitializing Direct Inversion Editor...")
    editor = DirectInversionEditor(
        device=device,
        num_ddim_steps=args.num_ddim_steps
    )
    
    # Create output directory
    output_dir = Path(args.output_path) / 'annotation_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    print("\nProcessing images...\n")
    success_count = 0
    error_count = 0
    
    for img_id, item in tqdm(filtered_annotations.items(), desc="Editing images"):
        try:
            # Get image path
            image_path = os.path.join(args.data_path, 'annotation_images', item['image_path'])
            
            # Get prompts
            prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
            prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
            blend_word = item.get('blended_word', None)
            
            # Create output path
            output_path = output_dir / item['image_path']
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run editing
            result = editor.edit_image(
                image_path=image_path,
                prompt_src=prompt_src,
                prompt_tar=prompt_tar,
                blend_word=blend_word,
                output_path=str(output_path)
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_id}: {str(e)}")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {args.output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

