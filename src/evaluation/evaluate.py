"""
Main evaluation script for PIE-Bench benchmark.
Evaluates editing results using SSIM, LPIPS, and CLIP similarity metrics.
"""

import json
import argparse
import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .metrics import MetricsCalculator, mask_decode


def evaluate_method(
    src_image_folder,
    tgt_image_folder,
    annotation_file,
    output_csv,
    edit_categories=None,
    device='cuda'
):
    """
    Evaluate a single editing method on PIE-Bench.
    
    Args:
        src_image_folder: Path to source images
        tgt_image_folder: Path to edited images
        annotation_file: Path to mapping_file.json
        output_csv: Path to save results CSV
        edit_categories: List of editing categories to evaluate (e.g., ['0', '1', '2'])
                        If None, evaluates all categories
        device: Device to run on ('cuda', 'mps', or 'cpu')
    """
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(device=device)
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Prepare results storage
    results = []
    
    # If no categories specified, evaluate all
    if edit_categories is None:
        edit_categories = [str(i) for i in range(10)]
    
    # Filter annotations by category
    filtered_annotations = {
        k: v for k, v in annotations.items()
        if v['editing_type_id'] in edit_categories
    }
    
    print(f"\nEvaluating {len(filtered_annotations)} images...")
    print(f"Categories: {edit_categories}")
    
    # Evaluate each image
    for img_id, item in tqdm(filtered_annotations.items(), desc="Evaluating"):
        try:
            # Get image paths
            base_image_path = item['image_path']
            src_image_path = os.path.join(src_image_folder, base_image_path)
            tgt_image_path = os.path.join(tgt_image_folder, base_image_path)
            
            # Check if files exist
            if not os.path.exists(src_image_path):
                print(f"Warning: Source image not found: {src_image_path}")
                continue
            if not os.path.exists(tgt_image_path):
                print(f"Warning: Target image not found: {tgt_image_path}")
                continue
            
            # Load images
            src_image = Image.open(src_image_path).convert('RGB')
            tgt_image = Image.open(tgt_image_path).convert('RGB')
            
            # Handle concatenated images (some methods output source+target side-by-side)
            if tgt_image.size[0] != tgt_image.size[1]:
                # Crop to get only the edited image (rightmost square)
                tgt_image = tgt_image.crop((
                    tgt_image.size[0] - 512,
                    tgt_image.size[1] - 512,
                    tgt_image.size[0],
                    tgt_image.size[1]
                ))
            
            # Get prompts
            src_prompt = item['original_prompt'].replace('[', '').replace(']', '')
            tgt_prompt = item['editing_prompt'].replace('[', '').replace(']', '')
            
            # Get mask
            mask = mask_decode(item['mask'])
            mask_3d = mask[:, :, np.newaxis].repeat(3, axis=2)
            
            # Calculate metrics
            # 1. SSIM - structural similarity (preservation)
            ssim = calculator.calculate_ssim(tgt_image, src_image)
            
            # 2. LPIPS - perceptual similarity (preservation)
            lpips = calculator.calculate_lpips(tgt_image, src_image)
            
            # 3. CLIP similarity - how well edited image matches target prompt
            clip_sim = calculator.calculate_clip_similarity(tgt_image, tgt_prompt)
            
            # Store results
            results.append({
                'image_id': img_id,
                'image_path': base_image_path,
                'editing_type': item['editing_type_id'],
                'ssim': ssim,
                'lpips': lpips,
                'clip_similarity': clip_sim,
            })
            
        except Exception as e:
            print(f"Error processing {img_id}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['image_id', 'image_path', 'editing_type', 'ssim', 'lpips', 'clip_similarity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # Calculate and print summary statistics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        ssim_mean = np.mean([r['ssim'] for r in results])
        lpips_mean = np.mean([r['lpips'] for r in results])
        clip_mean = np.mean([r['clip_similarity'] for r in results])
        
        print(f"\nTotal images evaluated: {len(results)}")
        print(f"\nAverage Metrics:")
        print(f"  SSIM:            {ssim_mean:.4f} (higher is better, max=1.0)")
        print(f"  LPIPS:           {lpips_mean:.4f} (lower is better, min=0.0)")
        print(f"  CLIP Similarity: {clip_mean:.4f} (higher is better)")
        
        # Per-category breakdown
        print(f"\nPer-Category Breakdown:")
        for cat in edit_categories:
            cat_results = [r for r in results if r['editing_type'] == cat]
            if cat_results:
                cat_ssim = np.mean([r['ssim'] for r in cat_results])
                cat_lpips = np.mean([r['lpips'] for r in cat_results])
                cat_clip = np.mean([r['clip_similarity'] for r in cat_results])
                print(f"  Category {cat} (n={len(cat_results):3d}): "
                      f"SSIM={cat_ssim:.4f}, LPIPS={cat_lpips:.4f}, CLIP={cat_clip:.4f}")
        
        print(f"\nResults saved to: {output_csv}")
        print("="*60)
        
    else:
        print("No results to save!")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate image editing methods on PIE-Bench'
    )
    parser.add_argument(
        '--src_image_folder',
        type=str,
        default='data/PIE-Bench_v1/annotation_images',
        help='Path to source images'
    )
    parser.add_argument(
        '--tgt_image_folder',
        type=str,
        required=True,
        help='Path to edited images (e.g., outputs/direct_inversion/annotation_images)'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default='data/PIE-Bench_v1/mapping_file.json',
        help='Path to mapping_file.json'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='results/evaluation_results.csv',
        help='Path to save evaluation results CSV'
    )
    parser.add_argument(
        '--edit_categories',
        nargs='+',
        type=str,
        default=None,
        help='Editing categories to evaluate (0-9). If not specified, evaluates all.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if needed
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
    
    # Run evaluation
    evaluate_method(
        src_image_folder=args.src_image_folder,
        tgt_image_folder=args.tgt_image_folder,
        annotation_file=args.annotation_file,
        output_csv=args.output_csv,
        edit_categories=args.edit_categories,
        device=device
    )


if __name__ == '__main__':
    main()

