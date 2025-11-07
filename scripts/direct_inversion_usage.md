# Direct Inversion - Quick Start

## Simple Commands (From Project Root)

### 1. Test on 1 Image (CPU, Fast)
```bash
conda run -n cis python scripts/test_direct_inversion_single.py --device cpu --num_steps 10
```

### 2. Run on 5 Sample Images (CPU)
```bash
conda run -n cis python scripts/run_pie_bench_sample.py --num_images 5 --device cpu --num_steps 10
```

### 3. Run on Full Category (CUDA Recommended)
```bash
# On CUDA (if available)
conda run -n cis python scripts/run_pie_bench_full.py --categories 0 --device cuda --num_steps 50

# On CPU (slower)
conda run -n cis python scripts/run_pie_bench_full.py --categories 0 --device cpu --num_steps 10
```

### 4. Evaluate Results
```bash
conda run -n cis python scripts/run_evaluation.py \
    --tgt_image_folder outputs/direct_inversion/annotation_images \
    --output_csv results/metrics.csv \
    --edit_categories 0
```

## All Options

### test_direct_inversion_single.py
```bash
--device {auto|cuda|cpu}    # Device to use (default: auto=cpu)
--num_steps N              # DDIM steps (default: 10)
```

### run_pie_bench_sample.py
```bash
--num_images N             # Number of images (default: 5)
--category C               # Category 0-9 (default: 0)
--device {auto|cuda|cpu}   # Device (default: auto=cpu)
--num_steps N              # DDIM steps (default: 10)
```

### run_pie_bench_full.py
```bash
--categories C1 C2 ...     # Categories to process (default: all 0-9)
--device {auto|cuda|cpu}   # Device (default: auto=cpu)
--num_steps N              # DDIM steps (default: 50)
```

**DDIM Steps:**
- **10 steps**: Fast, good for testing/CPU
- **20 steps**: Balanced
- **50 steps**: Best quality, use with CUDA

**Memory Issues?**
- Use `--device cpu` 
- Reduce `--num_steps` to 10 or less

## Output Structure

```
outputs/direct_inversion/annotation_images/
└── 0_random_140/*.jpg, 1_change_object_80/*.jpg, etc.

results/
└── metrics.csv  # SSIM, LPIPS, CLIP scores
```

