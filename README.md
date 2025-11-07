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

## Usage

See usage markdown in `scripts/` for different models.
