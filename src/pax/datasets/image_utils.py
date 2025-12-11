import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

def image_to_patches(tensor, patch_size=16):
    """
    Converts images to patch format for vision transformer processing.
    
    Args:
        tensor (torch.Tensor): Input image tensor of shape (N, C, H, W) where 
            N: batch size, C: channels, H: height, W: width.
        patch_size (int): Size of each square patch (default is 16).
    
    Returns:
        torch.Tensor: Tensor of patches with shape (N, num_patches, patch_dim)
            where num_patches = (H//patch_size) * (W//patch_size) and 
            patch_dim = patch_size * patch_size * C.
    """
    N, C, H, W = tensor.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 5, 1)
    patches = patches.contiguous().view(N, num_patches_h * num_patches_w, patch_size * patch_size * C)
    
    return patches

def patches_to_image(patches, patch_size=16, image_height=224, image_width=224):
    """
    Converts patch format back to image format.
    
    Args:
        patches (torch.Tensor): Patches tensor of shape (N, num_patches, patch_dim)
            where N: batch size, num_patches: total number of patches, 
            patch_dim: flattened patch dimension.
        patch_size (int): Size of each square patch (default is 16).
        image_height (int): Height of the output image (default is 224).
        image_width (int): Width of the output image (default is 224).
    
    Returns:
        torch.Tensor: Reconstructed image tensor of shape (N, C, H, W) where 
            N: batch size, C: channels, H: height, W: width.
    """
    N, num_patches, patch_dim = patches.shape
    channels = patch_dim // (patch_size * patch_size)
    num_patches_h = image_height // patch_size
    num_patches_w = image_width // patch_size
    
    patches = patches.view(N, num_patches_h, num_patches_w, 
                          patch_size, patch_size, channels)
    image = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    image = image.view(N, channels, image_height, image_width)
    
    return image