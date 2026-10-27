"""
Main logic for feature extraction.

Extracts ResNet50 features from CIFAR10 and saves them as a .mat file.
"""
import os
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from dataset import get_cifar10_dataloaders
from model import get_feature_extractor


def extract_features_from_loader(model, dataloader, device):
    """
    Extract features in batches from a DataLoader.

    Args:
        model: feature extraction model
        dataloader: data loader
        device: compute device
    Returns:
        features: feature matrix (N, 2048)
        labels:   label vector (N,)
    """
    all_features = []
    all_labels = []

    with torch.no_grad():  # No gradient computation; saves GPU memory
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def labels_to_onehot(labels, num_classes=10):
    """
    Convert class indices to one-hot encoding.

    Args:
        labels:       label vector (N,)
        num_classes:  number of classes
    Returns:
        onehot_labels: one-hot label matrix (N, num_classes)
    """
    onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        onehot[i, label] = 1.0
    return onehot


def save_features_to_mat(train_features, train_labels_onehot,
                          test_features, test_labels_onehot,
                          output_path):
    """
    Save features and labels to a .mat file.

    The format follows the mat structure used by traditional hashing datasets:
      - I_tr: training image features (N_train, 2048)
      - I_te: test image features     (N_test, 2048)
      - L_tr: training labels          (N_train, 10) one-hot
      - L_te: test labels              (N_test, 10) one-hot
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build the mat dictionary
    mat_dict = {
        'I_tr': train_features,                             # training image features
        'I_te': test_features,                              # test image features
        'L_tr': train_labels_onehot.astype(np.int32),       # training labels (one-hot)
        'L_te': test_labels_onehot.astype(np.int32),        # test labels (one-hot)
    }

    # Save as a .mat file
    sio.savemat(output_path, mat_dict)

    print(f"\nFeatures saved to: {output_path}")
    print(f"  Train feature shape: {train_features.shape}")
    print(f"  Test feature shape:  {test_features.shape}")
    print(f"  Train label shape:  {train_labels_onehot.shape}")
    print(f"  Test label shape:   {test_labels_onehot.shape}")


def run_feature_extraction(cfg):
    """Run the complete feature extraction pipeline."""
    # Set device
    device = torch.device(cfg['extraction']['device']
                          if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])

    # Load dataset
    print("\nLoading CIFAR10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = \
        get_cifar10_dataloaders(cfg)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Number of classes: {cfg['dataset']['num_classes']}")

    # Load model
    print("\nLoading ResNet50 model...")
    model = get_feature_extractor(cfg, device)
    print(f"Feature dimension: {model.feature_dim}")

    # Extract training set features
    print("\n=== Extracting training features ===")
    train_features, train_labels = extract_features_from_loader(
        model, train_loader, device
    )

    # Extract test set features
    print("\n=== Extracting test features ===")
    test_features, test_labels = extract_features_from_loader(
        model, test_loader, device
    )

    # Convert labels to one-hot encoding
    train_labels_onehot = labels_to_onehot(
        train_labels, cfg['dataset']['num_classes']
    )
    test_labels_onehot = labels_to_onehot(
        test_labels, cfg['dataset']['num_classes']
    )

    # Save as a .mat file
    print("\n=== Saving mat file ===")
    save_features_to_mat(
        train_features, train_labels_onehot,
        test_features, test_labels_onehot,
        cfg['output']['mat_path']
    )

    print("\n✅ Feature extraction complete!")
