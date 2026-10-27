"""
CIFAR10 dataset loading module.

Uses traditional data transformation methods.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(cfg):
    """
    Build train and test transforms from the config.

    Traditional transform pipeline:
        Resize -> CenterCrop/RandomCrop -> Flip -> ToTensor -> Normalize
    """
    train_cfg = cfg['transform']['train']
    test_cfg = cfg['transform']['test']

    # Training transforms (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(train_cfg['resize']),
        transforms.RandomCrop(train_cfg['crop_size']),
        transforms.RandomHorizontalFlip() if train_cfg['random_horizontal_flip'] else lambda x: x,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=train_cfg['mean'],
            std=train_cfg['std']
        )
    ])

    # Test transforms (no augmentation; center crop only)
    test_transform = transforms.Compose([
        transforms.Resize(test_cfg['resize']),
        transforms.CenterCrop(test_cfg['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=test_cfg['mean'],
            std=test_cfg['std']
        )
    ])

    return train_transform, test_transform


def get_cifar10_dataloaders(cfg):
    """
    Load the CIFAR10 dataset and return train/test DataLoaders.
    """
    data_root = cfg['dataset']['data_root']
    batch_size = cfg['extraction']['batch_size']
    num_workers = cfg['extraction']['num_workers']

    train_transform, test_transform = get_transforms(cfg)

    # Load training set
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )

    # Load test set
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling during feature extraction; keep original order
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader, train_dataset, test_dataset
