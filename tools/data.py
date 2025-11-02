import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict

from . import config

def get_transforms():

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN, 
            std=config.NORMALIZE_STD ) ])
    
    return transform

def load_dataset(data_path=None, transform=None, subset_size=None):

    if data_path is None:
        data_path = config.DATA_PATH
    if transform is None:
        transform = get_transforms()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    dataset = ImageFolder(root=data_path, transform=transform)
    
    if subset_size is not None:
        indices = list(range(min(subset_size, len(dataset))))
        dataset = Subset(dataset, indices)
        print(f"Using subset: {len(dataset)} images")
    else:
        print(f"Using full dataset: {len(dataset)} images")
    
    return dataset

def create_dataloader(dataset, batch_size=None, shuffle=False, 
                        num_workers=4, pin_memory=True):
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory )
    
    print(f"DataLoader ready: {len(loader)} batches, batch_size={batch_size}")
    return loader

def build_class_indices(dataset):
    
    class_indices = defaultdict(list)

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        for idx in dataset.indices:
            _, label = base_dataset.samples[idx]
            class_indices[label].append(idx)
    else:
        
        for idx, (_, label) in enumerate(dataset.samples):
            class_indices[label].append(idx)
    
    print(f"Total classes: {len(class_indices)}")
    print(f"Total samples: {sum(len(v) for v in class_indices.values())}")
    
    return class_indices

