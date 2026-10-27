# CIFAR10 ResNet50 Feature Extraction Tool

This is a teaching project for extracting ResNet50 deep features from the CIFAR10 dataset, producing the `.mat` feature files commonly used as baselines in traditional cross-modal hashing papers.

## 📁 Project Structure

```
cifar10_resnet50_feature_extraction/
├── configs/
│   └── default.yaml          # Config file (parameters, paths, model settings)
├── data/                     # Data storage directory
│   ├── cifar-10-batches-py/  # CIFAR10 raw data (auto-downloaded)
│   └── cifar10_resnet50_features.mat  # Extracted feature file
├── src/
│   ├── __init__.py
│   ├── dataset.py            # CIFAR10 dataset loading and traditional transforms
│   ├── model.py              # ResNet50 feature extraction model
│   └── extract_features.py   # Main feature extraction logic
├── run_extraction.py         # Main entry script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run feature extraction

```bash
python run_extraction.py --config configs/default.yaml
```

The CIFAR10 dataset (~170 MB) will be automatically downloaded on the first run.

## 📊 Output Description

The generated `cifar10_resnet50_features.mat` file contains the following fields:

| Field   | Shape         | Description                          |
|---------|---------------|--------------------------------------|
| `I_tr`  | (50000, 2048) | Training image features              |
| `I_te`  | (10000, 2048) | Test image features                  |
| `L_tr`  | (50000, 10)   | Training labels (one-hot)            |
| `L_te`  | (10000, 10)   | Test labels (one-hot)                |

## 🔧 Data Transforms

Uses **traditional image transforms** (adapted for ImageNet-pretrained models):

### Training transforms
1. Resize → 256
2. RandomCrop → 224×224
3. RandomHorizontalFlip
4. ToTensor
5. Normalize (ImageNet mean and std)

### Test transforms
1. Resize → 256
2. CenterCrop → 224×224
3. ToTensor
4. Normalize (ImageNet mean and std)

## 🎯 Model

- **Backbone**: ResNet50
- **Pretrained weights**: ImageNet pretrained
- **Feature output layer**: Global average pooling layer (the final FC classification layer is removed)
- **Feature dimension**: 2048

## ⚙️ Configuration

All parameters are configured in `configs/default.yaml`:

```yaml
seed: 42                    # Random seed (for reproducibility)
batch_size: 128             # Batch size
device: cuda                # Compute device
```

Simply run the script after modifying the config; no code changes are needed.

## 📝 Notes

1. **mat file format**: The `.mat` files produced by this project are mainly for compatibility with the baseline code of traditional hashing papers. For new projects, `.npz` or `.h5` formats are recommended.
2. **Feature dimension**: ResNet50 outputs 2048-dim features, which correspond to the channel count of ResNet50's last convolutional layer.
3. **Label format**: Saved as one-hot encoding (N, 10), consistent with traditional hashing dataset formats.
4. **Reproducibility**: The random seed is fixed; features extracted in the same environment should remain consistent.

## 🔍 Verify the mat file

After extraction, you can verify the result with:

```python
import scipy.io as sio

mat = sio.loadmat('data/cifar10_resnet50_features.mat')
print(mat.keys())
print('I_tr shape:', mat['I_tr'].shape)
print('L_te shape:', mat['L_te'].shape)
```

## 📚 References

- ResNet paper: *Deep Residual Learning for Image Recognition* (CVPR 2016)
- CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
