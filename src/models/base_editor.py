"""
Base class for all image editing methods.
Provides common interface for inversion and editing.
"""

from abc import ABC, abstractmethod
from PIL import Image
import torch


class BaseEditor(ABC):
    """
    Abstract base class for image editing methods.
    All editing methods should inherit from this class.
    """
    
    def __init__(self, device='cuda', num_ddim_steps=50):
        """
        Initialize the editor.
        
        Args:
            device: Device to run on ('cuda', 'mps', or 'cpu')
            num_ddim_steps: Number of DDIM steps for diffusion process
        """
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.model = None  # Will be initialized in subclasses
    
    @abstractmethod
    def edit_image(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        **kwargs
    ):
        """
        Edit an image based on source and target prompts.
        
        Args:
            image_path: Path to input image
            prompt_src: Source prompt describing the original image
            prompt_tar: Target prompt describing desired edit
            **kwargs: Method-specific parameters
            
        Returns:
            PIL.Image: Edited image
        """
        pass
    
    def __call__(self, *args, **kwargs):
        """Allow calling the editor like a function."""
        return self.edit_image(*args, **kwargs)

