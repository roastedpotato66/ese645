# Direct Inversion - Quick Start

## Simple Commands
**Try to run them on good GPUs!**
**I tried running them on my RTX3070ti (8g ram) even with num_steps = 5 will have OOM error**

### 1. Test on 1 Image
```bash
python scripts/test_direct_inversion_single.py --device cuda --num_steps 10
```

### 2. Run on 5 Sample Images
```bash
python scripts/run_pie_bench_sample.py --num_images 5 --device cuda --num_steps 10
```

### 3. Run on Full Category
```bash
# On CUDA (if available)
python scripts/run_pie_bench_full.py --categories 0 --device cuda --num_steps 50

# On CPU (slower)
python scripts/run_pie_bench_full.py --categories 0 --device cuda --num_steps 10
```

### 4. Evaluate Results
```bash
python scripts/run_evaluation.py \
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
- **10 steps**: Fast, good for testing
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

