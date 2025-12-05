#!/usr/bin/env python3
"""
Run Null-Text Inversion Variants (LB, Masa, LB+Masa) on PIE-Bench.
Skipping Baseline.
"""

import time
from datetime import timedelta
import argparse
import json
import sys
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from tqdm import tqdm

# Add parent directory to path
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

	if args.model == 'null_text':
    	model_kwargs["null_inner_steps"] = args.null_inner_steps
    	model_kwargs["null_lr"] = args.null_lr

	return ModelClass(**model_kwargs)

def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
    	torch.cuda.manual_seed_all(seed)
    
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
	model_choices = available_models()
	parser = argparse.ArgumentParser(description="Run Null-Text Inversion Variants on PIE-Bench")
    
	# Standard arguments
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--model', type=str, default='null_text', choices=model_choices)
	parser.add_argument('--model_id', type=str, default=None)
	parser.add_argument('--num_images', type=int, default=None)
	parser.add_argument('--category', type=str, nargs='+', default=['0'])
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
	parser.add_argument('--num_steps', type=int, default=50)
	parser.add_argument('--num_inversion_steps', type=int, default=None)
	parser.add_argument('--guidance_scale', type=float, default=None)
	parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp32')
	parser.add_argument('--verbose', action='store_true')
    
	# Null-Text specific arguments
	parser.add_argument('--null_inner_steps', type=int, default=10)
	parser.add_argument('--null_lr', type=float, default=0.01)

	# Variant Tuning Arguments
	parser.add_argument('--lb_steps', type=int, default=10, help='Number of steps for Latent Blending')
	parser.add_argument('--masa_start', type=int, default=5, help='Step to start MasaCtrl')

	# Parallel arguments
	parser.add_argument('--rank', type=int, default=0)
	parser.add_argument('--world_size', type=int, default=1)

	args = parser.parse_args()
    
	seed_everything(args.seed)

	# Detect device
	if args.device == 'auto':
    	if torch.cuda.is_available():
        	device = 'cuda'
    	else:
        	device = 'cpu'
	else:
    	device = args.device
    
	# Load annotations
	annotation_file = "data/PIE-Bench_v1/mapping_file.json"
	if not Path(annotation_file).exists():
    	print(f"Error: Annotation file not found at {annotation_file}")
    	sys.exit(1)

	with open(annotation_file, 'r') as f:
    	annotations = json.load(f)
    
	# Filter by category
	all_samples = [(k, v) for k, v in annotations.items() if v['editing_type_id'] in args.category]
    
	if args.num_images is not None:
    	all_samples = all_samples[:args.num_images]

	# Chunking
	total_samples = len(all_samples)
	chunk_size = math.ceil(total_samples / args.world_size)
	start_idx = args.rank * chunk_size
	end_idx = min(start_idx + chunk_size, total_samples)
	my_samples = all_samples[start_idx:end_idx]
    
	print(f"--- [GPU {args.rank}/{args.world_size}] Processing {len(my_samples)} images ---")
    
	if len(my_samples) == 0:
    	return

	# Initialize editor
	try:
    	editor = _build_editor(device, args)
	except Exception as e:
    	print(f"Failed to initialize model: {e}")
    	sys.exit(1)
    
	# --- Define Variants (Removed Baseline) ---
	variants: List[Tuple[str, Dict[str, Any]]] = [
    	# 1. Latent Blending Only
    	(
        	"lb",  	 
        	{
            	"use_masactrl": False,
            	"use_latent_blending": True,
            	"latent_blend_steps": args.lb_steps
        	}
    	),
    	# 2. MasaCtrl Only
    	(
        	"masa",	 
        	{
            	"use_masactrl": True,  
            	"use_latent_blending": False,
            	"masactrl_step_start": args.masa_start
        	}
    	),
    	# 3. Both
    	(
        	"lb_masa",  
        	{
            	"use_masactrl": True,  
            	"use_latent_blending": True,
            	"latent_blend_steps": args.lb_steps,
            	"masactrl_step_start": args.masa_start
        	}
    	)
	]

	# Process images
	total = len(my_samples)
	progress = tqdm(my_samples, desc=f"GPU {args.rank}", unit="img", position=0)
	start_time = time.perf_counter()
	processed_counts = 0
    
	for idx, (img_id, item) in enumerate(progress, start=1):
    	progress.set_description(f"[{idx}/{total}] {img_id}")
   	 
    	image_path = f"data/PIE-Bench_v1/annotation_images/{item['image_path']}"
    	prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
    	prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
    	blend_word = item.get('blended_word', None)
   	 
    	for variant_name, variant_config in variants:
        	output_root = Path("outputs") / f"{args.model}_{variant_name}"
        	output_path = output_root / "annotation_images" / item['image_path']
        	output_path.parent.mkdir(parents=True, exist_ok=True)
       	 
        	if output_path.exists():
            	continue

        	# Update Config
        	for k, v in variant_config.items():
            	if hasattr(editor.config, k):
                	setattr(editor.config, k, v)

        	try:
            	editor.edit_image(
                	image_path=image_path,
                	prompt_src=prompt_src,
                	prompt_tar=prompt_tar,
                	blend_word=blend_word,
                	output_path=str(output_path),
                	verbose=args.verbose
            	)
        	except Exception as e:
            	progress.write(f"Error on {img_id} [{variant_name}]: {e}")
            	continue
   	 
    	processed_counts += 1

	progress.close()
	elapsed = time.perf_counter() - start_time
	print(f"GPU {args.rank} Finished! Processed {processed_counts} images in {timedelta(seconds=elapsed)}")

if __name__ == '__main__':
	main()



