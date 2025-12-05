#!/usr/bin/env python3
"""
Run a suite of benchmark configurations and collect evaluation metrics.
"""

import subprocess
import re
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


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
    # More flexible pattern that handles various whitespace
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


def main():
    # Global settings
    common_args = [
        '--device', 'cuda',
        '--num_steps', '50',
        '--num_inversion_steps', '50',
        '--num_images', '50',
        '--category', '0'
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
    
    for model in models:
        print(f"\n\n{'*'*80}")
        print(f"Processing Model: {model.upper()}")
        print(f"{'*'*80}")
        
        for config in configurations:
            config_id = config['id']
            config_label = config['label']
            config_args = config['args']
            
            full_label = f"[{model}] {config_label}"
            
            print(f"\n\n{'#'*80}")
            print(f"# Run {config_id}/3 for {model}: {config_label}")
            print(f"{'#'*80}\n")
            
            # Build command for run_pie_bench_sample.py
            # Add model argument specifically
            run_cmd = ['python', 'scripts/run_pie_bench_sample.py'] + common_args + ['--model', model] + config_args
            
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
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'error': True
                })
                continue
            
            # Run evaluation
            # We need to point to the specific output directory for this model
            # outputs/{model}/annotation_images
            eval_output_dir = f"outputs/{model}/annotation_images"
            
            # Unique CSV for this run
            result_csv_name = f"metrics_{model}_conf{config_id}.csv"
            result_csv_path = f"results/{result_csv_name}"
            
            eval_cmd = [
                'python', 'scripts/run_evaluation.py',
                '--tgt_image_folder', eval_output_dir,
                '--output_csv', result_csv_path
            ]
            
            stdout, stderr, returncode = run_command(
                eval_cmd,
                f"Evaluation: {full_label}"
            )
            
            if returncode != 0:
                print(f"\n[ERROR] Evaluation failed with return code {returncode}")
                print(f"[ERROR] stderr: {stderr[:500]}")  # Print first 500 chars of stderr
                results.append({
                    'model': model,
                    'config_id': config_id,
                    'label': config_label,
                    'args': ' '.join(config_args),
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'error': True
                })
                continue
            
            # Parse metrics from evaluation output (combine stdout and stderr, use CSV as fallback)
            metrics = parse_metrics(stdout, stderr, result_csv_path)
            
            # Validate that we got at least some metrics
            if not metrics:
                print(f"\n[WARNING] Could not parse any metrics from evaluation output")
                print(f"[WARNING] Attempting to read from CSV: {result_csv_path}")
                # Try one more time with just CSV
                if Path(result_csv_path).exists():
                    metrics = parse_metrics("", "", result_csv_path)
            
            if not metrics:
                print(f"\n[ERROR] Failed to extract metrics. Marking as error.")
                results.append({
                    'model': model,
                    'config_id': config_id,
                    'label': config_label,
                    'args': ' '.join(config_args),
                    'ssim': None,
                    'lpips': None,
                    'clip_similarity': None,
                    'error': True
                })
                continue
            
            result = {
                'model': model,
                'config_id': config_id,
                'label': config_label,
                'args': ' '.join(config_args),
                'ssim': metrics.get('ssim'),
                'lpips': metrics.get('lpips'),
                'clip_similarity': metrics.get('clip_similarity'),
                'error': False
            }
            
            results.append(result)
            
            print(f"\n[SUCCESS] {full_label} completed:")
            print(f"  SSIM: {metrics.get('ssim', 'N/A')}")
            print(f"  LPIPS: {metrics.get('lpips', 'N/A')}")
            print(f"  CLIP: {metrics.get('clip_similarity', 'N/A')}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"results/benchmark_suite_results_{timestamp}.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Suite Results\n\n")
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
        
        f.write("\n## Detailed Configuration Arguments\n\n")
        for result in results:
            f.write(f"### [{result['model']}] Config {result['config_id']}: {result['label']}\n\n")
            f.write(f"**Arguments:** `{result['args'] if result['args'] else '(none)'}`\n\n")
            if result['error']:
                f.write("**Status:** ERROR\n\n")
            else:
                f.write(f"**Metrics:**\n")
                f.write(f"- SSIM: {result['ssim']:.4f}\n" if result['ssim'] else "- SSIM: N/A\n")
                f.write(f"- LPIPS: {result['lpips']:.4f}\n" if result['lpips'] else "- LPIPS: N/A\n")
                f.write(f"- CLIP: {result['clip_similarity']:.4f}\n" if result['clip_similarity'] else "- CLIP: N/A\n")
            f.write("\n")
    
    # Also save as CSV
    csv_file = Path(f"results/benchmark_suite_results_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("model,config_id,label,args,ssim,lpips,clip_similarity\n")
        for result in results:
            ssim = str(result['ssim']) if result['ssim'] is not None else "N/A"
            lpips = str(result['lpips']) if result['lpips'] is not None else "N/A"
            clip = str(result['clip_similarity']) if result['clip_similarity'] is not None else "N/A"
            args = result['args'].replace(',', ';') if result['args'] else ""
            f.write(f"{result['model']},{result['config_id']},{result['label']},{args},{ssim},{lpips},{clip}\n")
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUITE COMPLETE")
    print('='*80)
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {csv_file}")

if __name__ == '__main__':
    main()
