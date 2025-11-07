# ESE 6450 Final Project: Text-Driven Image Editing

**Team:** Anbang Chen, Shizhuo Mu, Nora Han

## Meeting Summary
### Nov 2nd
* By Nov 7th, create github repo, prepare data and evaluation metrices.
* By Nov 15th, ready for baseline methods and begin to discuss how to improve models.
* Presentation time is Dec 2nd 9:45AM.

## Repository Structure

```
ese645/
├── src/
│   ├── models/          # Baseline implementations
│   ├── evaluation/      # Metrics (SSIM, LPIPS, CLIP)
│   └── utils/           # Image processing utilities
├── scripts/             # Run scripts
├── outputs/             # Generated images
├── results/             # Evaluation CSVs
└── Documentation (README, guides)
```

## Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n ese645 python=3.9 -y
conda activate ese645

# Install PyTorch...

# Install other dependencies
pip install -r requirements.txt
```

### 2. Data

The PIE-Bench dataset is already downloaded in `data/PIE-Bench_v1/`:
- 700 images across 10 editing types
- Annotations in `mapping_file.json`
- Images in `annotation_images/`

## Usage

### Run Direct Inversion Baseline

```bash
# Process a few images for testing (on MPS)
python scripts/run_direct_inversion.py \
    --edit_categories 0 \
    --num_images 5 \
    --device mps \
    --num_ddim_steps 20

# Process all images in category 0 (on CUDA)
python scripts/run_direct_inversion.py \
    --edit_categories 0 \
    --device cuda \
    --num_ddim_steps 50

# Process multiple categories
python scripts/run_direct_inversion.py \
    --edit_categories 0 1 2 \
    --device cuda
```

**Note:** First run will download Stable Diffusion v1.4 model (~4GB).

### Run Evaluation

```bash
# Evaluate Direct Inversion results
python scripts/run_evaluation.py \
    --tgt_image_folder outputs/direct_inversion/annotation_images \
    --output_csv results/direct_inversion_metrics.csv \
    --edit_categories 0 \
    --device cuda
```

**Output:** CSV file with SSIM, LPIPS, and CLIP similarity for each image.

### Test Evaluation Metrics Only

```bash
# Test metrics calculator on random images
cd src/evaluation
python metrics.py
```
