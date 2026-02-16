import os
import json
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from image_utils import image_to_patches, patches_to_image


def load_images(
    image_sources,
    apply_patches=False,
    patch_size=16,
    target_size=(224, 224),
    return_labels=False,
    label_column='label',
    dataset_base_path='/'
):
    """
    Generalized function to load images from various sources.
    
    Args:
        image_sources (str, list, or dict): Can be:
            - Single image path/URL (str)
            - List of image paths/URLs
            - Dataset name (str) - loads from CSV metadata
            - Dict with 'dataset_name' and optional 'metadata_path'
        apply_patches (bool): Whether to apply image_to_patches transformation
        patch_size (int): Size of patches for transformation
        target_size (tuple): Target size to resize images to
        return_labels (bool): Whether to return labels (only for dataset loading)
        label_column (str): Column name for labels in metadata CSV
        dataset_base_path (str): Base path for dataset loading
    
    Returns:
        torch.Tensor or tuple: 
            - Single image: tensor of shape (C, H, W) or (num_patches, patch_dim)
            - Multiple images: tensor of shape (N, C, H, W) or (N, num_patches, patch_dim)
            - If return_labels=True: (images_tensor, labels_tensor)
    """
    if apply_patches:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: image_to_patches(x.unsqueeze(0), patch_size).squeeze(0))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    
    def _load_single_image(image_path):
        """Helper to load a single image."""
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        return transform(img)
    
    # Handle different input types
    if isinstance(image_sources, str):
        # Check if it's a single image path/URL or dataset name
        if image_sources.startswith(('http://', 'https://')) or os.path.isfile(image_sources):
            # Single image path or URL
            return _load_single_image(image_sources)
        else:
            # Assume it's a dataset name
            return _load_dataset_images(
                image_sources, transform, return_labels, label_column, dataset_base_path
            )
    
    elif isinstance(image_sources, dict):
        # Dataset specification with custom parameters
        dataset_name = image_sources.get('dataset_name')
        metadata_path = image_sources.get('metadata_path')
        if metadata_path is None:
            metadata_path = f'{dataset_base_path}/{dataset_name}/metadata.csv'
        
        return _load_dataset_images_from_metadata(
            metadata_path, dataset_base_path, dataset_name, transform, return_labels, label_column
        )
    
    elif isinstance(image_sources, list):
        # List of image paths/URLs
        images = []
        for path in tqdm(image_sources, desc="Loading images"):
            img_tensor = _load_single_image(path)
            images.append(img_tensor)
        
        return torch.stack(images)
    
    else:
        raise ValueError(f"Unsupported image_sources type: {type(image_sources)}")


def _load_dataset_images(dataset_name, transform, return_labels, label_column, dataset_base_path):
    """Helper to load images from dataset using standard metadata CSV."""
    metadata_path = f'{dataset_base_path}/{dataset_name}/metadata.csv'
    return _load_dataset_images_from_metadata(
        metadata_path, dataset_base_path, dataset_name, transform, return_labels, label_column
    )


def _load_dataset_images_from_metadata(metadata_path, dataset_base_path, dataset_name, transform, return_labels, label_column):
    """Helper to load images from metadata CSV file."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    image_paths = metadata['image_path'].tolist()
    
    desc = f"Loading images from {dataset_name}" if dataset_name else "Loading images"
    tensors = []
    labels = []
    
    for idx, image_filename in enumerate(tqdm(image_paths, desc=desc)):
        image_full_path = f'{dataset_base_path}/{dataset_name}/{image_filename}' if dataset_name else image_filename
        image = Image.open(image_full_path).convert("RGB")
        image_tensor = transform(image)
        tensors.append(image_tensor)
        
        if return_labels:
            if label_column in metadata.columns:
                labels.append(metadata.iloc[idx][label_column])
            else:
                labels.append(0) 
    
    images_tensor = torch.stack(tensors, dim=0)
    
    if return_labels:
        labels_tensor = torch.tensor(labels)
        print(f"Loaded {len(images_tensor)} images as tensor of shape {images_tensor.shape} with {len(labels_tensor)} labels.")
        return images_tensor, labels_tensor
    else:
        print(f"Loaded {len(images_tensor)} images as tensor of shape {images_tensor.shape}.")
        return images_tensor


def load_image(image_path, apply_patches=False, patch_size=16, target_size=(224, 224)):
    """Load a single image (backward compatibility)."""
    return load_images(image_path, apply_patches, patch_size, target_size)


def load_images_tensor(dataset_name='Broden-Pascal', apply_patches=False, patch_size=16, base_path='/Data'):
    """Load images from dataset (backward compatibility)."""
    return load_images(dataset_name, apply_patches, patch_size, dataset_base_path=base_path)


def load_images_tensor_with_labels(dataset_name='Broden-Pascal', label_column='label', apply_patches=False, patch_size=16, base_path='/Data'):
    """Load images with labels from dataset (backward compatibility)."""
    return load_images(
        dataset_name, 
        apply_patches=apply_patches, 
        patch_size=patch_size, 
        return_labels=True,
        label_column=label_column,
        dataset_base_path=base_path
    )


def load_multiple_images(image_paths, apply_patches=False, patch_size=16, target_size=(224, 224)):
    """Load multiple images from paths (backward compatibility)."""
    return load_images(image_paths, apply_patches, patch_size, target_size)