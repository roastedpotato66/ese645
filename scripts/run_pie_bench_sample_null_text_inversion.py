#!/usr/bin/env python3
"""
Ablation Test on PIE-Bench (Category 0, First 3 images).
Tests:
1. Latent Blending Only
2. MasaCtrl Only
3. Combined

Output Structure:
  outputs/ablation_test/
    ├── 01_blend_only/
    │   ├── image1.png
    │   └── ...
    ├── 02_masa_only/
    │   └── ...
    └── 03_combined/
        └── ...
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm

# Add parent directory to path to find src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your updated NullTextEditor
from src.models.registry import available_models, get_model_builder

# Setup Logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/ablation_test.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w" 
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run PIE-Bench Ablation Test")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--base_steps', type=int, default=50, help='Total inference steps')
    args = parser.parse_args()

    # 1. Setup Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"=== Starting Ablation Test on {device} ===")

    # 2. Define the 3 Experimental Configurations
    experiments = {
        "01_blend_only": {
            "use_latent_blending": True,
            "latent_blend_steps": 15,    # 前30%步数锁死
            "use_masactrl": False,
        },
        "02_masa_only": {
            "use_latent_blending": False,
            "use_masactrl": True,
            "masactrl_step_start": 3,    # 全程开启 (Mask 会保护物体不被抹除)
        },
        "03_combined": {
            "use_latent_blending": True,
            "latent_blend_steps": 5,    # 前20%步数锁死背景
            "use_masactrl": True,
            "masactrl_step_start": 10,   # Latent Blending 结束后 MasaCtrl 接手
        }
    }

    # 3. Initialize the Editor
    print("Initializing NullTextEditor...")
    
    # 全局配置：注入 MasaCtrl 需要的层关键字
    global_config = {
        "masactrl_layer_keywords": ["up_blocks.1", "up_blocks.2", "up_blocks.3"]
    }
    
    try:
        ModelClass = get_model_builder("null_text")
        editor = ModelClass(
            device=device,
            model_id="runwayml/stable-diffusion-v1-5", 
            num_inference_steps=args.base_steps,
            num_inversion_steps=args.base_steps,
            null_inner_steps=10,
            null_lr=0.01,
            precision='fp32',
            config_overrides=global_config 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 4. Load PIE-Bench Data
    annotation_file = "data/PIE-Bench_v1/mapping_file.json"
    if not Path(annotation_file).exists():
        print(f"Error: {annotation_file} not found. Please run from project root.")
        sys.exit(1)

    with open(annotation_file, 'r') as f:
        full_annotations = json.load(f)

    # 5. Filter: Category 0, First 3 images
    target_samples = []
    count = 0
    target_category = "0" 
    
    for key, item in full_annotations.items():
        if item.get('editing_type_id') == target_category:
            target_samples.append((key, item))
            count += 1
            if count >= 3:
                break
    
    if len(target_samples) == 0:
        print(f"Warning: No images found for Category {target_category}. Check mapping file format.")
        sys.exit(1)

    print(f"Selected {len(target_samples)} images from Category {target_category} for testing.")

    # 6. Run Execution Loop
    for img_id, item in target_samples:
        print(f"\nProcessing Image ID: {img_id}")
        
        image_rel_path = item['image_path']
        image_path = f"data/PIE-Bench_v1/annotation_images/{image_rel_path}"
        
        prompt_src = item['original_prompt'].replace('[', '').replace(']', '')
        prompt_tar = item['editing_prompt'].replace('[', '').replace(']', '')
        blend_word = item.get('blended_word', None)
        
        # === 【修改点 1】提取 Mask ===
        mask_rle = item.get('mask', None)

        print(f"  Src: {prompt_src}")
        print(f"  Tar: {prompt_tar}")
        if mask_rle:
            print(f"  Mask found (len={len(mask_rle)})")

        # Run all 3 experiments for this image
        for exp_folder_name, exp_kwargs in experiments.items():
            print(f"  > Mode: {exp_folder_name} ...")
            
            experiment_dir = Path("outputs") / "ablation_test" / exp_folder_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = Path(image_rel_path).name
            output_path = experiment_dir / output_filename
            
            if output_path.exists():
                print(f"    Skipping (Exists): {output_path}")
                continue

            try:
                editor.edit_image(
                    image_path=image_path,
                    prompt_src=prompt_src,
                    prompt_tar=prompt_tar,
                    blend_word=blend_word,
                    
                    # === 【修改点 2】传入 Mask ===
                    provided_mask=mask_rle,
                    
                    output_path=str(output_path),
                    verbose=False,
                    **exp_kwargs 
                )
            except Exception as e:
                print(f"    FAILED {exp_folder_name}: {e}")
                logger.error(f"Failed {img_id} - {exp_folder_name}: {e}")

    print("\n=== Ablation Test Complete ===")
    print(f"Results saved in: outputs/ablation_test/")
    for exp in experiments.keys():
        print(f"  - outputs/ablation_test/{exp}/")

if __name__ == '__main__':
    main()