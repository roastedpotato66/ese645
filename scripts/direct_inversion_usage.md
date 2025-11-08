# Direct Inversion - Quick Start

## Simple Commands

### 1. Test on 1 Image
```bash
python scripts/test_direct_inversion_single.py --device cuda --num_steps 10
```

### 2. Run on 5 Sample Images
```bash
python scripts/run_pie_bench_sample.py --num_images 5 --device cuda --num_steps 10
```

### 3. Run Full PIE-Bench (All 10 Categories)
```bash
# Recommended CUDA run (30 DDIM steps fits on A100; use 10–15 on smaller GPUs)
# Batch size is automatically detected based on GPU memory
python scripts/run_pie_bench_full.py \
    --categories 0 1 2 3 4 5 6 7 8 9 \
    --device cuda \
    --num_steps 20 \
    --batch_size auto

# Or manually specify batch size (e.g., for A100 80GB, try batch_size=10)
python scripts/run_pie_bench_full.py \
    --categories 0 1 2 3 4 5 6 7 8 9 \
    --device cuda \
    --num_steps 20 \
    --batch_size 10
```

### 4. Evaluate Full PIE-Bench Results
```bash
# Metrics: SSIM / LPIPS / CLIP
python scripts/run_evaluation.py \
    --tgt_image_folder outputs/direct_inversion/annotation_images \
    --output_csv results/direct_inversion_full.csv \
    --edit_categories 0 1 2 3 4 5 6 7 8 9 \
    --device cuda
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
--batch_size {auto|N}      # Batch size (default: auto=detect automatically)
```

### run_pie_bench_full.py
```bash
--categories C1 C2 ...     # Categories to process (default: all 0-9)
--device {auto|cuda|cpu}   # Device (default: auto=cpu)
--num_steps N              # DDIM steps (default: 50)
--batch_size {auto|N}      # Batch size (default: auto=detect automatically)
```

**DDIM Steps:**
- **10 steps**: Fast, good for testing
- **20 steps**: Balanced
- **50 steps**: Best quality, use with CUDA

**Batch Processing (Current Implementation):**
- Batch size is automatically detected based on GPU memory
- **Current Status**: Images are processed in batches for memory management
- **Limitation**: UNet calls are sequential (per-image attention controllers)
- **Speedup**: ~1.1-1.3x from batched VAE/text encoding (not full parallelization)
- **Future**: True UNet batching requires refactoring attention controllers
- **A100 (80GB)**: Batch size ~10-12 (organizes processing, limited speedup)
- **A100 (40GB)**: Batch size ~6-8 (organizes processing, limited speedup)
- **L4/RTX 4090 (24GB)**: Batch size ~3-4 (organizes processing, limited speedup)
- **RTX 3070 Ti (8GB)**: Batch size 1 (no batching, sequential)
- Override with `--batch_size N` to manually set batch size

**Memory Issues?**
- Use `--device cpu` 
- Reduce `--num_steps` to 10 or less
- Reduce `--batch_size` to 1 (disable batching)

## Output Structure

```
outputs/direct_inversion/annotation_images/
└── 0_random_140/*.jpg, 1_change_object_80/*.jpg, etc.

results/
└── metrics.csv  # SSIM, LPIPS, CLIP scores
```

