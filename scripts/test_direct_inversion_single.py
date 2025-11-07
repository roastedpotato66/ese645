#!/usr/bin/env python3
"""
Simple test script for Direct Inversion on 1 sample image.
"""

import os
import sys
import argparse
from pathlib import Path

# Parse args first to set environment before importing torch
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='auto', 
                   choices=['auto', 'cuda', 'mps', 'cpu'],
                   help='Device to use (default: auto)')
parser.add_argument('--num_steps', type=int, default=10,
                   help='Number of DDIM steps (default: 10, use fewer for CPU)')
args = parser.parse_args()

# Force CPU environment if specified
if args.device == 'cpu' or args.device == 'auto':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices

import torch
import json

# Add parent directory to path (since we're in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.direct_inversion import DirectInversionEditor

def main():
    # Detect device (args already parsed above)
    if args.device == 'auto':
        # Only use CUDA if explicitly available, default to CPU
        device = 'cuda' if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"DDIM steps: {args.num_steps}")
    
    # Load one sample from PIE-Bench
    data_path = "data/PIE-Bench_v1"
    annotation_file = f"{data_path}/mapping_file.json"
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Get first image from category 0
    sample_id = None
    for img_id, item in annotations.items():
        if item['editing_type_id'] == '0':
            sample_id = img_id
            sample_item = item
            break
    
    if sample_id is None:
        print("No sample found in category 0!")
        return
    
    # Get image details
    image_path = f"{data_path}/annotation_images/{sample_item['image_path']}"
    prompt_src = sample_item['original_prompt'].replace('[', '').replace(']', '')
    prompt_tar = sample_item['editing_prompt'].replace('[', '').replace(']', '')
    blend_word = sample_item.get('blended_word', None)
    
    print(f"\n{'='*60}")
    print(f"Testing Direct Inversion")
    print(f"{'='*60}")
    print(f"Sample ID: {sample_id}")
    print(f"Image: {image_path}")
    print(f"Source prompt: {prompt_src}")
    print(f"Target prompt: {prompt_tar}")
    print(f"Blend word: {blend_word}")
    print(f"{'='*60}\n")
    
    # Initialize editor
    print("Initializing Direct Inversion Editor...")
    editor = DirectInversionEditor(
        device=device,
        num_ddim_steps=args.num_steps
    )
    
    # Run editing
    print("\nRunning Direct Inversion...")
    result = editor.edit_image(
        image_path=image_path,
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        blend_word=blend_word,
        output_path=f"outputs/test_direct_inversion_{sample_id}.png",
        return_intermediate=True
    )
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"Output saved to: outputs/test_direct_inversion_{sample_id}.png")
    print(f"{'='*60}")
    
    # Show result info
    if isinstance(result, dict):
        print("\nResult images:")
        print(f"  - Source: {result['source'].size}")
        print(f"  - Reconstructed: {result['reconstructed'].size}")
        print(f"  - Edited: {result['edited'].size}")

if __name__ == '__main__':
    main()

