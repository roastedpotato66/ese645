"""
Device utilities for automatic batch size detection and GPU memory management.
"""

import torch


def get_gpu_memory_info(device='cuda'):
    """
    Get GPU memory information.
    
    Args:
        device: Device string (e.g., 'cuda', 'cuda:0')
        
    Returns:
        dict: Dictionary with total_memory_gb, free_memory_gb, used_memory_gb
    """
    if not torch.cuda.is_available() or device == 'cpu':
        return {
            'total_memory_gb': 0,
            'free_memory_gb': 0,
            'used_memory_gb': 0,
            'available': False
        }
    
    try:
        device_obj = torch.device(device)
        device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_memory_gb = props.total_memory / (1024 ** 3)
        
        # Get current memory usage
        torch.cuda.reset_peak_memory_stats(device_index)
        used_memory_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        free_memory_gb = total_memory_gb - used_memory_gb
        
        return {
            'total_memory_gb': total_memory_gb,
            'free_memory_gb': free_memory_gb,
            'used_memory_gb': used_memory_gb,
            'available': True
        }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return {
            'total_memory_gb': 0,
            'free_memory_gb': 0,
            'used_memory_gb': 0,
            'available': False
        }


def get_optimal_batch_size(device='cuda', num_ddim_steps=50, model_memory_gb=5.0, safety_factor=0.8):
    """
    Heuristic-based batch size estimation.
    
    This function estimates the optimal batch size based on:
    - GPU memory capacity
    - Model memory footprint
    - Number of DDIM steps (more steps = more memory per image)
    - Safety factor to leave headroom
    
    Args:
        device: Device string (e.g., 'cuda', 'cuda:0')
        num_ddim_steps: Number of DDIM steps (affects memory per image)
        model_memory_gb: Estimated memory for model weights (default: 5GB for SD v1.4)
        safety_factor: Safety factor to leave headroom (default: 0.8 = 80% usage)
        
    Returns:
        int: Recommended batch size (at least 1)
    """
    # No batching on CPU
    if device == 'cpu' or not torch.cuda.is_available():
        return 1
    
    # Get GPU memory info
    mem_info = get_gpu_memory_info(device)
    if not mem_info['available']:
        return 1
    
    total_memory_gb = mem_info['total_memory_gb']
    
    # Estimate available memory after model loading
    # Model is already loaded, so we use current usage as baseline
    # But we still need to account for model weights in our calculation
    available_memory_gb = total_memory_gb - model_memory_gb
    
    # Per-image memory estimate
    # Base memory per image: ~4GB for 50 steps
    # This scales approximately linearly with number of steps
    base_mem_per_image_gb = 4.0
    step_factor = 1.0 + (num_ddim_steps - 50) * 0.02  # Scale factor for steps
    mem_per_image_gb = base_mem_per_image_gb * step_factor
    
    # Calculate batch size with safety factor
    # Leave some headroom for activations, gradients (if any), etc.
    usable_memory_gb = available_memory_gb * safety_factor
    batch_size = int(usable_memory_gb / mem_per_image_gb)
    
    # Clamp to reasonable bounds
    # Minimum: 1 (no batching)
    # Maximum: 16 (sanity check, very large batches have diminishing returns)
    batch_size = max(1, min(batch_size, 16))
    
    return batch_size


def get_batch_size_with_override(device='cuda', num_ddim_steps=50, batch_size_override=None):
    """
    Get batch size with optional override.
    
    Args:
        device: Device string
        num_ddim_steps: Number of DDIM steps
        batch_size_override: Optional override ('auto', int, str, or None)
                           - 'auto' or None: Use automatic detection
                           - int or str(int): Use specified batch size
                           
    Returns:
        int: Batch size to use
    """
    # Handle None or 'auto'
    if batch_size_override is None or (isinstance(batch_size_override, str) and batch_size_override.lower() == 'auto'):
        return get_optimal_batch_size(device, num_ddim_steps)
    
    # Handle integer
    if isinstance(batch_size_override, int):
        return max(1, batch_size_override)  # Ensure at least 1
    
    # Try to parse as int (handles string numbers like "4", "10")
    if isinstance(batch_size_override, str):
        try:
            return max(1, int(batch_size_override))
        except (ValueError, TypeError):
            print(f"Warning: Invalid batch_size_override '{batch_size_override}', using automatic detection")
            return get_optimal_batch_size(device, num_ddim_steps)
    
    # Fallback to automatic detection
    print(f"Warning: Invalid batch_size_override type '{type(batch_size_override)}', using automatic detection")
    return get_optimal_batch_size(device, num_ddim_steps)


def print_batch_size_info(device, num_ddim_steps, batch_size, mem_info=None):
    """
    Print batch size information for debugging.
    
    Args:
        device: Device string
        num_ddim_steps: Number of DDIM steps
        batch_size: Selected batch size
        mem_info: Optional memory info dict (if None, will fetch)
    """
    if mem_info is None:
        mem_info = get_gpu_memory_info(device)
    
    print(f"\n{'='*60}")
    print(f"Batch Size Configuration")
    print(f"{'='*60}")
    if mem_info['available']:
        print(f"GPU Memory: {mem_info['total_memory_gb']:.1f} GB total, "
              f"{mem_info['used_memory_gb']:.1f} GB used, "
              f"{mem_info['free_memory_gb']:.1f} GB free")
    else:
        print(f"Device: {device} (no GPU memory info available)")
    print(f"DDIM Steps: {num_ddim_steps}")
    print(f"Batch Size: {batch_size}")
    if batch_size > 1:
        print(f"  → Processing images in batches of {batch_size}")
        print(f"  → Note: VAE/text encoding is batched (~10-20% speedup)")
        print(f"  → Note: UNet calls are sequential (per-image attention controllers)")
        print(f"  → Expected speedup: ~1.1-1.3x (limited by sequential UNet)")
    else:
        print(f"  → Sequential processing (no batching)")
    print(f"{'='*60}\n")

