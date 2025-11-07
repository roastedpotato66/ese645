#!/usr/bin/env python3
"""
Evaluate editing results using SSIM, LPIPS, and CLIP metrics.

Usage:
    python scripts/run_evaluation.py --tgt_image_folder outputs/direct_inversion/annotation_images
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the evaluation script
from src.evaluation.evaluate import main

if __name__ == '__main__':
    main()

