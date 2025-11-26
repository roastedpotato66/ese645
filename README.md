# ESE 6450 Final Project: Text-Driven Image Editing

**Team:** Anbang Chen, Shizhuo Mu, Nora Han

## Meeting Summary
### Nov 2nd
* By Nov 7th, create github repo, prepare data and evaluation metrices.
* By Nov 15th, ready for baseline methods and begin to discuss how to improve models.
* Presentation time is Dec 2nd 9:45AM.

### Nov 25th
* Anbang finished ddim; Shizhuo finished null_text_inversion.
* Nora will fix the bug in direct inversion.
* Everyone should propose an extension method and discuss the feasibility on Nov 26th 11AM.

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
├── notebooks/           # For Colab
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

## Example Usage

### Run Single Test

```
python scripts/test_ddim_single.py --device cuda --num_steps 30 --num_inversion_steps 50
```
```
 python scripts/test_null_text_inversion.py  --device cuda --num_steps 30  --num_inversion_steps 30 --null_inner_steps 10  --null_lr 0.01
```
```
python scripts/test_direct_inversion.py --device cuda --num_steps 30
```

### Run Sample Images from PIE-Bench
```
python scripts/run_pie_bench_sample.py --model ddim --device cuda --num_steps 50 --num_inversion_steps 50 --num_images 5 --category 0
```

### Run All Images from PIE-Bench

```
python scripts/run_pie_bench_full.py --model ddim --device cuda --num_steps 50 --num_inversion_steps 50 --categories 0 1 2 3 4 5 6 7 8 9
```

### Evaluate Generated Results and Save Metrics
```
python scripts/run_evaluation.py --tgt_image_folder outputs/ddim/annotation_images --output_csv results/ddim_metrics.csv
```

