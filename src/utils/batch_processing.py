"""
Batch processing utilities for image editing models.

This module provides reusable utilities for batch processing that can be used
by any editor implementation. It handles:
- Automatic batch size detection
- Batched VAE encoding/decoding
- Batched text encoding
- Progress tracking
- Memory management

Note: UNet batching is method-specific and requires per-method implementation
due to attention controllers and other method-specific requirements.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Callable, Tuple
from src.utils.device_utils import get_batch_size_with_override, get_gpu_memory_info
from src.utils.image_utils import images2latent_batch, latent2image, load_512


def batch_encode_images(vae, image_paths, verbose=False):
    """
    Batch encode multiple images to latents using VAE.
    
    This is a reusable utility that any editor can use for batched VAE encoding.
    
    Args:
        vae: VAE model from Stable Diffusion
        image_paths: List of image paths or numpy arrays
        verbose: Whether to print progress
        
    Returns:
        List of latent tensors, one per image
    """
    if verbose:
        print(f"Batch encoding {len(image_paths)} images...")
    
    # Load all images
    images = []
    for img_path in image_paths:
        if isinstance(img_path, str):
            images.append(load_512(img_path))
        elif isinstance(img_path, np.ndarray):
            images.append(img_path)
        else:
            images.append(np.array(img_path))
    
    # Batch encode
    try:
        batch_latents = images2latent_batch(vae, images)
        # Split back to individual latents
        latents = [batch_latents[i:i+1] for i in range(len(image_paths))]
        if verbose:
            print(f"✓ Batch encoded {len(image_paths)} images")
        return latents
    except Exception as e:
        if verbose:
            print(f"Warning: Batch encode failed, falling back to individual: {e}")
        # Fallback to individual encoding
        from src.utils.image_utils import image2latent
        latents = []
        for img in images:
            latents.append(image2latent(vae, img))
        return latents


def batch_encode_text(text_encoder, tokenizer, prompts, device, verbose=False):
    """
    Batch encode multiple text prompts using text encoder.
    
    This is a reusable utility that any editor can use for batched text encoding.
    
    Args:
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        List of text embeddings, one per prompt
    """
    if verbose:
        print(f"Batch encoding {len(prompts)} prompts...")
    
    try:
        # Batch encode all prompts
        text_input = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch_text_embeddings = text_encoder(
            text_input.input_ids.to(device)
        )[0].to(dtype=text_encoder.dtype)
        
        # Split back to individual embeddings
        embeddings = [batch_text_embeddings[i:i+1] for i in range(len(prompts))]
        
        if verbose:
            print(f"✓ Batch encoded {len(prompts)} prompts")
        return embeddings
    except Exception as e:
        if verbose:
            print(f"Warning: Batch text encode failed: {e}")
        # Fallback to individual encoding
        embeddings = []
        for prompt in prompts:
            text_input = tokenizer(
                [prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            emb = text_encoder(text_input.input_ids.to(device))[0]
            embeddings.append(emb)
        return embeddings


def process_batches(
    items: List,
    batch_size: int,
    process_fn: Callable,
    progress_callback: Optional[Callable] = None,
    verbose: bool = False
):
    """
    Generic batch processing utility.
    
    This can be used by any editor to process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_fn: Function to process a batch of items
                    Signature: process_fn(batch_items) -> results
        progress_callback: Optional callback for progress updates
                          Signature: progress_callback(completed, total, item_id)
        verbose: Whether to print verbose output
        
    Returns:
        List of results
    """
    results = []
    total = len(items)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = items[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\nProcessing batch {batch_num}/{total_batches} "
                  f"(items {batch_start+1}-{batch_end} of {total})")
        
        try:
            batch_results = process_fn(batch_items)
            results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                for i, item in enumerate(batch_items):
                    item_id = str(item) if not isinstance(item, (list, tuple)) else str(item[0])
                    progress_callback(batch_start + i + 1, total, item_id)
                    
        except Exception as e:
            if verbose:
                print(f"Error processing batch {batch_num}: {e}")
            # Add None results for failed batch
            results.extend([None] * len(batch_items))
            continue
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


class BatchProcessor:
    """
    Reusable batch processor for image editing models.
    
    This class provides a framework for batch processing that teammates can extend
    for their own models. It handles:
    - Automatic batch size detection
    - Memory management
    - Progress tracking
    - Error handling
    
    Usage:
        processor = BatchProcessor(device='cuda', num_steps=50)
        results = processor.process_images(
            image_paths=paths,
            process_fn=my_editor.edit_image,
            batch_size='auto'
        )
    """
    
    def __init__(self, device='cuda', num_ddim_steps=50, batch_size='auto'):
        """
        Initialize batch processor.
        
        Args:
            device: Device to run on
            num_ddim_steps: Number of DDIM steps (for batch size estimation)
            batch_size: Batch size ('auto' for automatic, or integer)
        """
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.batch_size = get_batch_size_with_override(
            device=device,
            num_ddim_steps=num_ddim_steps,
            batch_size_override=batch_size
        )
    
    def process_images(
        self,
        image_paths: List[str],
        process_fn: Callable,
        process_kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable] = None,
        verbose: bool = False
    ) -> List:
        """
        Process images in batches.
        
        Args:
            image_paths: List of image paths
            process_fn: Function to process a single image
                       Signature: process_fn(image_path, **process_kwargs) -> result
            process_kwargs: Additional kwargs to pass to process_fn
            progress_callback: Optional progress callback
            verbose: Whether to print verbose output
            
        Returns:
            List of results
        """
        if process_kwargs is None:
            process_kwargs = {}
        
        results = []
        total = len(image_paths)
        
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_paths = image_paths[batch_start:batch_end]
            
            if verbose:
                print(f"Processing batch: {batch_start+1}-{batch_end} of {total}")
            
            # Process batch
            batch_results = []
            for i, image_path in enumerate(batch_paths):
                try:
                    result = process_fn(image_path, **process_kwargs)
                    batch_results.append(result)
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(batch_start + i + 1, total, image_path)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error processing {image_path}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

