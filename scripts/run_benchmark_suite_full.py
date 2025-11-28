#!/usr/bin/env python3
"""
Run a suite of benchmark configurations on the full PIE-Bench dataset and collect evaluation metrics.
Includes category-wise analysis and plotting.
"""

import subprocess
import re
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


def run_command(cmd: List[str], description: str) -> Tuple[str, str, int]:
    """Run a command and return stdout, stderr, and return code."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.stdout, result.stderr, result.returncode


def parse_metrics(eval_output: str, eval_stderr: str = "", csv_path: str = None) -> Dict[str, float]:
    """Parse SSIM, LPIPS, and CLIP metrics from evaluation output."""
    metrics = {}
    
    # Combine stdout and stderr (sometimes output goes to stderr)
    combined_output = eval_output + "\n" + eval_stderr
    
    # Pattern to match: "  SSIM:            0.1234 (higher is better, max=1.0)"
    ssim_pattern = r'SSIM:\s+([\d.]+)'
    lpips_pattern = r'LPIPS:\s+([\d.]+)'
    clip_pattern = r'CLIP Similarity:\s+([\d.]+)'
    
    ssim_match = re.search(ssim_pattern, combined_output, re.IGNORECASE)
    lpips_match = re.search(lpips_pattern, combined_output, re.IGNORECASE)
    clip_match = re.search(clip_pattern, combined_output, re.IGNORECASE)
    
    if ssim_match:
        try:
            metrics['ssim'] = float(ssim_match.group(1))
        except ValueError:
            pass
    
    if lpips_match:
        try:
            metrics['lpips'] = float(lpips_match.group(1))
        except ValueError:
            pass
    
    if clip_match:
        try:
            metrics['clip_similarity'] = float(clip_match.group(1))
        except ValueError:
            pass
    
    # If we didn't find all metrics in output, try reading from CSV as fallback
    if csv_path and Path(csv_path).exists():
        if not metrics.get('ssim') or not metrics.get('lpips') or not metrics.get('clip_similarity'):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Calculate averages from CSV
                        ssim_vals = [float(r['ssim']) for r in rows if 'ssim' in r and r['ssim']]
                        lpips_vals = [float(r['lpips']) for r in rows if 'lpips' in r and r['lpips']]
                        clip_vals = [float(r['clip_similarity']) for r in rows if 'clip_similarity' in r and r['clip_similarity']]
                        
                        if ssim_vals and not metrics.get('ssim'):
                            metrics['ssim'] = sum(ssim_vals) / len(ssim_vals)
                        if lpips_vals and not metrics.get('lpips'):
                            metrics['lpips'] = sum(lpips_vals) / len(lpips_vals)
                        if clip_vals and not metrics.get('clip_similarity'):
                            metrics['clip_similarity'] = sum(clip_vals) / len(clip_vals)
            except Exception as e:
                print(f"[WARNING] Could not read metrics from CSV {csv_path}: {e}")
    
    return metrics


def calculate_category_metrics(csv_path: str) -> Dict[str, Dict[str, float]]:
    """Calculate average metrics per category from CSV file."""
    category_metrics = defaultdict(lambda: {'ssim': [], 'lpips': [], 'clip_similarity': []})
    
    if not Path(csv_path).exists():
        return {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row.get('editing_type', 'unknown')
                if 'ssim' in row and row['ssim']:
                    try:
                        category_metrics[category]['ssim'].append(float(row['ssim']))
                    except ValueError:
                        pass
                if 'lpips' in row and row['lpips']:
                    try:
                        category_metrics[category]['lpips'].append(float(row['lpips']))
                    except ValueError:
                        pass
                if 'clip_similarity' in row and row['clip_similarity']:
                    try:
                        category_metrics[category]['clip_similarity'].append(float(row['clip_similarity']))
                    except ValueError:
                        pass
        
        # Calculate averages
        result = {}
        for category, metrics in category_metrics.items():
            result[category] = {
                'ssim': np.mean(metrics['ssim']) if metrics['ssim'] else None,
                'lpips': np.mean(metrics['lpips']) if metrics['lpips'] else None,
                'clip_similarity': np.mean(metrics['clip_similarity']) if metrics['clip_similarity'] else None,
                'count': len(metrics['ssim'])
            }
        return result
    except Exception as e:
        print(f"[WARNING] Could not calculate category metrics from {csv_path}: {e}")
        return {}


def create_plots(results: List[Dict], output_dir: Path):
    """Create plots for metrics across configurations and categories."""
    # Prepare data for plotting
    models = sorted(set(r['model'] for r in results if not r.get('error')))
    configs = sorted(set(r['config_id'] for r in results if not r.get('error')))
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Benchmark Suite Results - Full PIE-Bench', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('ssim', 'SSIM (Higher is Better)', axes[0]),
        ('lpips', 'LPIPS (Lower is Better)', axes[1]),
        ('clip_similarity', 'CLIP Similarity (Higher is Better)', axes[2])
    ]
    
    for metric_name, ylabel, ax in metrics_to_plot:
        # Prepare data: model x config matrix
        data_matrix = []
        labels = []
        
        for model in models:
            for config_id in configs:
                result = next((r for r in results if r['model'] == model and r['config_id'] == config_id and not r.get('error')), None)
                if result:
                    value = result.get(metric_name)
                    data_matrix.append(value if value is not None else 0)
                    labels.append(f"{model}\nConfig {config_id}")
                else:
                    data_matrix.append(0)
                    labels.append(f"{model}\nConfig {config_id}\n(Error)")
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, data_matrix, alpha=0.7, edgecolor='black')
        
        # Color bars by model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (bar, result) in enumerate(zip(bars, [next((r for r in results if r['model'] == models[i // len(configs)] and r['config_id'] == configs[i % len(configs)] and not r.get('error')), None) for i in range(len(labels))])):
            if result:
                bar.set_color(colors[i % len(colors)])
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(ylabel, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'benchmark_metrics_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved comparison plot to: {plot_path}")
    plt.close()
    
    # Create category-wise comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Category-wise Metrics Comparison', fontsize=16, fontweight='bold')
    
    for metric_idx, (metric_name, ylabel, ax) in enumerate(metrics_to_plot):
        # Collect category data for each configuration
        x = np.arange(len(categories))
        width = 0.15  # Width of bars
        
        config_data = {}
        for result in results:
            if result.get('error') or not result.get('category_metrics'):
                continue
            key = f"{result['model']}_conf{result['config_id']}"
            config_data[key] = []
            for cat in categories:
                cat_metrics = result['category_metrics'].get(cat, {})
                config_data[key].append(cat_metrics.get(metric_name, 0))
        
        # Plot bars for each configuration
        offset = -width * (len(config_data) - 1) / 2
        for i, (key, values) in enumerate(sorted(config_data.items())):
            ax.bar(x + offset, values, width, label=key.replace('_', ' '), alpha=0.8)
            offset += width
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(ylabel, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'benchmark_category_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved category comparison plot to: {plot_path}")
    plt.close()


def main():
    # Global settings
    common_args = [
        '--device', 'cuda',
        '--num_steps', '50',
        '--num_inversion_steps', '50',
    ]
    
    models = ['ddim', 'direct_inversion']
    
    # Define 3 configurations per model (6 total runs)
    configurations = [
        {
            'id': 1,
            'label': 'Latent Blending',
            'args': ['--use_latent_blending', '--latent_blend_steps', '15']
        },
        {
            'id': 2,
            'label': 'MasaCtrl',
            'args': ['--use_masactrl', '--masactrl_step_start', '40', '--masactrl_layer_keywords', 'up_blocks.1', 'up_blocks.2']
        },
        {
            'id': 3,
            'label': 'Latent Blending + MasaCtrl',
            'args': [
                '--use_latent_blending', '--latent_blend_steps', '15',
                '--use_masactrl', '--masactrl_step_start', '40', '--masactrl_layer_keywords', 'up_blocks.1', 'up_blocks.2'
            ]
        },
    ]
    
    results = []
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in models:
        print(f"\n\n{'*'*80}")
        print(f"Processing Model: {model.upper()}")
        print(f"{'*'*80}")
        
        for config in configurations:
            config_id = config['id']
            config_label = config['label']
            config_args = config['args']
            
            full_label = f"[{model}] {config_label}"
            
            # Create unique output directory for this run
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = f"outputs/{model}_{config_id}_{run_timestamp}"
            output_dir = f"{output_base}/annotation_images"
            
            print(f"\n\n{'#'*80}")
            print(f"# Run {config_id}/3 for {model}: {config_label}")
            print(f"# Output: {output_dir}")
            print(f"{'#'*80}\n")
            
            # Build command for run_pie_bench_full.py
            # Pass the base directory, script will append /annotation_images
            run_cmd = ['python', 'scripts/run_pie_bench_full.py'] + common_args + ['--model', model, '--output_path', output_base] + config_args
            
            # Run the benchmark
            stdout, stderr, returncode = run_command(
                run_cmd,
                f"Benchmark Run: {full_label}"
            )
            
            if returncode != 0:
                print(f"\n[ERROR] Benchmark run failed with return code {returncode}")
                results.append({
                    'model': model,
                    'config_id': config_id,
                    'label': config_label,
                    'args': ' '.join(config_args),
                    'output_dir': output_dir,
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'category_metrics': {},
                    'error': True
                })
                continue
            
            # Run evaluation
            result_csv_name = f"metrics_{model}_conf{config_id}_{run_timestamp}.csv"
            result_csv_path = f"results/{result_csv_name}"
            
            eval_cmd = [
                'python', 'scripts/run_evaluation.py',
                '--tgt_image_folder', output_dir,
                '--output_csv', result_csv_path
            ]
            
            stdout, stderr, returncode = run_command(
                eval_cmd,
                f"Evaluation: {full_label}"
            )
            
            if returncode != 0:
                print(f"\n[ERROR] Evaluation failed with return code {returncode}")
                print(f"[ERROR] stderr: {stderr[:500]}")
                results.append({
                    'model': model,
                    'config_id': config_id,
                    'label': config_label,
                    'args': ' '.join(config_args),
                    'output_dir': output_dir,
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'category_metrics': {},
                    'error': True
                })
                continue
            
            # Parse overall metrics from evaluation output
            metrics = parse_metrics(stdout, stderr, result_csv_path)
            
            # Calculate category-wise metrics
            category_metrics = calculate_category_metrics(result_csv_path)
            
            # Validate that we got at least some metrics
            if not metrics:
                print(f"\n[WARNING] Could not parse any metrics from evaluation output")
                print(f"[WARNING] Attempting to read from CSV: {result_csv_path}")
                if Path(result_csv_path).exists():
                    metrics = parse_metrics("", "", result_csv_path)
            
            if not metrics:
                print(f"\n[ERROR] Failed to extract metrics. Marking as error.")
                results.append({
                    'model': model,
                    'config_id': config_id,
                    'label': config_label,
                    'args': ' '.join(config_args),
                    'output_dir': output_dir,
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'category_metrics': {},
                    'error': True
                })
                continue
            
            result = {
                'model': model,
                'config_id': config_id,
                'label': config_label,
                'args': ' '.join(config_args),
                'output_dir': output_dir,
                'ssim': metrics.get('ssim'),
                'lpips': metrics.get('lpips'),
                'clip_similarity': metrics.get('clip_similarity'),
                'category_metrics': category_metrics,
                'error': False
            }
            
            results.append(result)
            
            print(f"\n[SUCCESS] {full_label} completed:")
            print(f"  SSIM: {metrics.get('ssim', 'N/A')}")
            print(f"  LPIPS: {metrics.get('lpips', 'N/A')}")
            print(f"  CLIP: {metrics.get('clip_similarity', 'N/A')}")
            print(f"  Categories evaluated: {len(category_metrics)}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"results/benchmark_suite_full_results_{timestamp}.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Suite Results (Full PIE-Bench)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Config ID | Configuration | SSIM | LPIPS | CLIP Similarity |\n")
        f.write("|-------|-----------|---------------|------|-------|-----------------|\n")
        
        for result in results:
            model = result['model']
            config_id = result['config_id']
            label = result['label']
            ssim = f"{result['ssim']:.4f}" if result['ssim'] is not None else "N/A"
            lpips = f"{result['lpips']:.4f}" if result['lpips'] is not None else "N/A"
            clip = f"{result['clip_similarity']:.4f}" if result['clip_similarity'] is not None else "N/A"
            
            if result['error']:
                ssim = lpips = clip = "ERROR"
            
            f.write(f"| {model} | {config_id} | {label} | {ssim} | {lpips} | {clip} |\n")
        
        f.write("\n## Category-wise Metrics\n\n")
        for result in results:
            if result['error'] or not result.get('category_metrics'):
                continue
            
            f.write(f"### [{result['model']}] Config {result['config_id']}: {result['label']}\n\n")
            f.write("| Category | SSIM | LPIPS | CLIP Similarity | Count |\n")
            f.write("|----------|------|-------|-----------------|-------|\n")
            
            for category in sorted(result['category_metrics'].keys()):
                cat_metrics = result['category_metrics'][category]
                ssim = f"{cat_metrics['ssim']:.4f}" if cat_metrics['ssim'] is not None else "N/A"
                lpips = f"{cat_metrics['lpips']:.4f}" if cat_metrics['lpips'] is not None else "N/A"
                clip = f"{cat_metrics['clip_similarity']:.4f}" if cat_metrics['clip_similarity'] is not None else "N/A"
                count = cat_metrics.get('count', 0)
                f.write(f"| {category} | {ssim} | {lpips} | {clip} | {count} |\n")
            f.write("\n")
        
        f.write("\n## Detailed Configuration Arguments\n\n")
        for result in results:
            f.write(f"### [{result['model']}] Config {result['config_id']}: {result['label']}\n\n")
            f.write(f"**Arguments:** `{result['args'] if result['args'] else '(none)'}`\n\n")
            f.write(f"**Output Directory:** `{result['output_dir']}`\n\n")
            if result['error']:
                f.write("**Status:** ERROR\n\n")
            else:
                f.write(f"**Metrics:**\n")
                f.write(f"- SSIM: {result['ssim']:.4f}\n" if result['ssim'] else "- SSIM: N/A\n")
                f.write(f"- LPIPS: {result['lpips']:.4f}\n" if result['lpips'] else "- LPIPS: N/A\n")
                f.write(f"- CLIP: {result['clip_similarity']:.4f}\n" if result['clip_similarity'] else "- CLIP: N/A\n")
            f.write("\n")
    
    # Also save as CSV
    csv_file = Path(f"results/benchmark_suite_full_results_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("model,config_id,label,args,output_dir,ssim,lpips,clip_similarity\n")
        for result in results:
            ssim = str(result['ssim']) if result['ssim'] is not None else "N/A"
            lpips = str(result['lpips']) if result['lpips'] is not None else "N/A"
            clip = str(result['clip_similarity']) if result['clip_similarity'] is not None else "N/A"
            args = result['args'].replace(',', ';') if result['args'] else ""
            f.write(f"{result['model']},{result['config_id']},{result['label']},{args},{result['output_dir']},{ssim},{lpips},{clip}\n")
    
    # Create plots
    print(f"\n{'='*80}")
    print("Creating plots...")
    print('='*80)
    create_plots(results, Path("results"))
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUITE COMPLETE")
    print('='*80)
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {csv_file}")
    print(f"\nPlots saved to:")
    print(f"  - results/benchmark_metrics_comparison.png")
    print(f"  - results/benchmark_category_comparison.png")


if __name__ == '__main__':
    main()

