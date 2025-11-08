"""
Evaluation metrics for image editing quality assessment.
Supports SSIM, LPIPS, and CLIP similarity metrics.
Extracted and adapted from PnPInversion evaluation code.
"""

import torch
import numpy as np
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class MetricsCalculator:
    """
    Calculator for image editing evaluation metrics.
    Supports SSIM, LPIPS, and CLIP similarity.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize metrics calculators.
        
        Args:
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        self.device = device
        
        # Handle MPS device compatibility
        # Note: Some metrics may not work on MPS, fallback to CPU if needed
        metric_device = device if device in ['cuda', 'cpu'] else 'cpu'
        
        print(f"Initializing metrics on device: {metric_device}")
        
        # CLIP metric for text-image similarity
        self.clip_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(metric_device)
        
        # SSIM metric for structural similarity
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(metric_device)
        
        # LPIPS metric for perceptual similarity
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type='squeeze'
        ).to(metric_device)
        
        self.metric_device = metric_device
    
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        """
        Calculate SSIM (Structural Similarity Index) between two images.
        Higher is better (range: -1 to 1, typically 0 to 1).
        
        Args:
            img_pred: Predicted/edited image (PIL Image or numpy array)
            img_gt: Ground truth/source image (PIL Image or numpy array)
            mask_pred: Optional mask for predicted image
            mask_gt: Optional mask for ground truth image
            
        Returns:
            float: SSIM score
        """
        # Convert to numpy and normalize to [0, 1]
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        
        # Apply masks if provided
        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
        
        # Convert to torch tensors [1, C, H, W]
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.metric_device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.metric_device)
        
        # Calculate SSIM
        score = self.ssim_metric(img_pred_tensor, img_gt_tensor)
        return score.cpu().item()
    
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity).
        Lower is better (range: 0 to 1+).
        
        Args:
            img_pred: Predicted/edited image (PIL Image or numpy array)
            img_gt: Ground truth/source image (PIL Image or numpy array)
            mask_pred: Optional mask for predicted image
            mask_gt: Optional mask for ground truth image
            
        Returns:
            float: LPIPS score
        """
        # Convert to numpy and normalize to [0, 1]
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        
        # Apply masks if provided
        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
        
        # Convert to torch tensors [1, C, H, W]
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.metric_device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.metric_device)
        
        # LPIPS expects images in range [-1, 1]
        score = self.lpips_metric(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        return score.cpu().item()
    
    def calculate_clip_similarity(self, img, txt, mask=None):
        """
        Calculate CLIP similarity between image and text.
        Higher is better.
        
        Args:
            img: Image (PIL Image or numpy array)
            txt: Text prompt (string)
            mask: Optional mask to focus on specific regions
            
        Returns:
            float: CLIP similarity score
        """
        # Convert to numpy array if needed
        img = np.array(img)
        
        # Apply mask if provided
        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)
        
        # Convert to torch tensor [C, H, W]
        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.metric_device)
        
        # Calculate CLIP score
        score = self.clip_metric(img_tensor, txt)
        return score.cpu().item()
    
    def calculate_all_metrics(self, src_image, tgt_image, src_prompt, tgt_prompt, 
                             mask=None, return_dict=True):
        """
        Calculate all metrics at once for convenience.
        
        Args:
            src_image: Source/original image
            tgt_image: Target/edited image
            src_prompt: Source prompt
            tgt_prompt: Target prompt
            mask: Optional mask for region-specific evaluation
            return_dict: If True, return dict; else return tuple
            
        Returns:
            dict or tuple: Metrics (SSIM, LPIPS, CLIP_target)
        """
        ssim = self.calculate_ssim(tgt_image, src_image, mask, mask)
        lpips = self.calculate_lpips(tgt_image, src_image, mask, mask)
        clip_sim = self.calculate_clip_similarity(tgt_image, tgt_prompt, mask)
        
        if return_dict:
            return {
                'ssim': ssim,
                'lpips': lpips,
                'clip_similarity': clip_sim
            }
        else:
            return ssim, lpips, clip_sim


def mask_decode(encoded_mask, image_shape=(512, 512)):
    """
    Decode RLE-encoded mask from PIE-Bench annotations.
    
    Args:
        encoded_mask: RLE-encoded mask (list of integers)
        image_shape: Output image shape (H, W)
        
    Returns:
        numpy.ndarray: Decoded binary mask
    """
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    # Decode RLE
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    
    # Set boundary to 1 to avoid annotation errors
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    
    return mask_array


if __name__ == "__main__":
    # Simple test
    print("Testing metrics calculator...")
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    calculator = MetricsCalculator(device=device)
    
    # Create dummy images for testing
    img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    print("\nCalculating metrics on random images...")
    ssim = calculator.calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    lpips = calculator.calculate_lpips(img1, img2)
    print(f"LPIPS: {lpips:.4f}")
    
    clip_sim = calculator.calculate_clip_similarity(img1, "a test image")
    print(f"CLIP Similarity: {clip_sim:.4f}")
    
    print("\nMetrics calculator test completed!")

