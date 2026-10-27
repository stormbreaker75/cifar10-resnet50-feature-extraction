"""
CIFAR10 ResNet50 Feature Extraction Main Entry
Usage: python run_extraction.py --config configs/default.yaml
"""
import argparse
import yaml
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.extract_features import run_feature_extraction


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10 ResNet50 Feature Extraction')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to the config file'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()

    # Load config
    print(f"Loading config file: {args.config}")
    cfg = load_config(args.config)

    # Run feature extraction
    run_feature_extraction(cfg)


if __name__ == '__main__':
    main()
