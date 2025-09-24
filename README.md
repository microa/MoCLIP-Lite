# MoCLIP-Lite: Efficient Video Recognition by Fusing CLIP with Motion Vectors

A lightweight video action recognition framework that combines Motion Vectors (MV) with CLIP for efficient video understanding on the UCF-101 dataset.

## 🎯 Overview

MoCLIP-Lite leverages compressed video motion vectors as a lightweight alternative to traditional RGB-based video analysis, combined with CLIP's powerful text-image understanding capabilities for zero-shot action recognition.

## ✨ Key Features

- **Motion Vector Processing**: Efficient extraction and processing of motion vectors from compressed videos
- **Multi-modal Fusion**: Late fusion of motion vector features with CLIP text embeddings
- **Zero-shot Learning**: CLIP-based zero-shot action recognition capabilities
- **Lightweight Architecture**: Based on EfficientNet for efficient inference
- **Comprehensive Evaluation**: Support for both supervised and zero-shot evaluation

## 🏗️ Architecture

The framework consists of three main components:

1. **MV-TSN Model**: Motion Vector Temporal Segment Network based on EfficientNet
2. **CLIP Integration**: Pre-trained CLIP model for text-image understanding
3. **Late Fusion**: Combines MV features with CLIP text embeddings for final classification

## 📋 Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
timm>=0.6.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
PIL>=8.3.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/microa/MoCLIP-Lite.git
cd MoCLIP-Lite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package (optional, for development):
```bash
pip install -e .
```

## 📊 Dataset Preparation

1. Download UCF-101 dataset and extract to your data directory
2. Extract motion vectors from videos
3. Update data paths in the configuration files

## 🔧 Usage

### Training MV-TSN Model

```bash
python training/train_mv_tsn.py --data_root /path/to/ucf101 --mv_root /path/to/motion_vectors
```

### Training Fusion Model

```bash
python training/train_fusion.py --mv_model_path mv_tsn_best_model.pth --data_root /path/to/ucf101
```

### Zero-shot Evaluation

```bash
python evaluation/evaluate_zeroshot.py --data_root /path/to/ucf101 --mv_root /path/to/motion_vectors
```

### Testing Final Fusion Model

```bash
python evaluation/test_fusion_final.py --model_path fusion_best_model.pth --data_root /path/to/ucf101
```

## 📁 Project Structure

```
MoCLIP-Lite/
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── configs/                     # Configuration files
│   ├── __init__.py
│   ├── class_mappings.json      # Class name mappings
│   └── prompt_templates.json    # Text prompt templates
├── data/                        # Data processing modules
│   ├── __init__.py
│   ├── dataloader.py            # Basic data loader
│   ├── dataloader_coviar.py     # TSN data loader
│   └── transforms_video.py      # Video transformations
├── models/                      # Model definitions
│   ├── __init__.py
│   └── model.py                 # MV-TSN model definition
├── training/                    # Training scripts
│   ├── __init__.py
│   ├── train_mv_tsn.py          # MV model training
│   ├── train_fusion.py          # Fusion model training
│   └── train_mv_only.py         # MV-only training
├── evaluation/                  # Evaluation and testing
│   ├── __init__.py
│   ├── evaluate_zeroshot.py     # Zero-shot evaluation
│   ├── test_fusion_final.py     # Fusion model testing
│   ├── test_mv_tsn.py           # MV model testing
│   └── test_mv_tsn_32.py        # MV model testing (32 segments)
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── mv_quiver.py             # Motion vector visualization
│   ├── mv_to_rgb.py             # MV to RGB conversion
│   ├── generate_text_features.py    # Text feature generation
│   └── precompute_clip_features.py  # CLIP feature precomputation
└── scripts/                     # Example scripts
    ├── __init__.py
    └── example_usage.py         # Usage examples
```

## 🎨 Visualization

The framework includes tools for visualizing motion vectors and generating qualitative analysis reports:

```bash
python utils/mv_quiver.py --input_dir /path/to/motion_vectors --output_dir /path/to/visualizations
```

## 📈 Results

The model achieves competitive performance on UCF-101 dataset with significantly reduced computational requirements compared to RGB-based methods.

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{huang2025mocliplite,
  title={MoCLIP-Lite: Efficient Video Recognition by Fusing CLIP with Motion Vectors},
  author={Huang, Binhua and Wang, Nan and Parakash, Arjun and Dev, Soumyabrata},
  journal={arXiv preprint arXiv:2509.17084},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCF-101 dataset creators
- CLIP model by OpenAI
- EfficientNet by Google Research
- PyTorch and the open-source community
