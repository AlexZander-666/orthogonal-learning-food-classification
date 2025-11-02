# Orthogonal Learning: Complementary Biases of Knowledge Distillation and Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Orthogonal Learning Framework**: Demonstrating how knowledge distillation and lightweight attention mechanisms achieve complementary improvements through functionally orthogonal inductive biases.

## 🎯 Overview

This repository implements a systematic study of lightweight attention mechanisms and knowledge distillation on mobile-first architectures for fine-grained visual recognition. Our key finding: attention and distillation provide **nearly perfectly additive gains** (4.27% ≈ 1.63% + 2.68%), mathematically confirming they operate through independent, orthogonal mechanisms.

### 🌟 Key Highlights

- ✅ **Orthogonal Learning Theory**: First to interpret complementary effects as synergy of two orthogonal inductive biases
- ✅ **Strong Empirical Results**: MobileNetV3 student (78.50%) surpasses ResNet-50 teacher (76.76%) with only 21.5% parameters
- ✅ **Parameter Efficiency**: ECA (~500 params) and SimAM (zero params) outperform heavyweight mechanisms
- ✅ **Cross-Domain Validation**: Consistent relative improvements on Food-101 (+4.27%) and Flowers-102 (+2.33%)
- ✅ **Comprehensive Analysis**: Ablation studies, statistical significance tests, hyperparameter interaction analysis

### 📊 Main Results

| Model | Parameters | FLOPs | Accuracy | Notes |
|-------|-----------|-------|----------|-------|
| ResNet-50 (Teacher) | 25.6M | 4.1G | 76.76% | Teacher model |
| MobileNetV3-Large (Baseline) | 5.5M | 0.22G | 74.23% | Baseline |
| **MobileNetV3 + ECA + KD** | **5.5M** | **0.22G** | **78.50%** | **Our method (recommended)** |
| **MobileNetV3 + SimAM + KD** | **5.5M** | **0.22G** | **78.12%** | **Our method (zero-param)** |

> 🎉 Student model achieves **1.74 pp higher accuracy** than teacher with **only 21.5% of its parameters**!

---

## 📁 Project Structure

```
orthogonal-learning-food-classification/
├── models/                      # Model implementations
│   ├── attention_modules.py    # Attention mechanisms (ECA, SimAM, CBAM, SE, CoordAtt)
│   └── mobilenetv3_attention.py # MobileNetV3 + Attention
├── experiments/                 # Experiment scripts
│   ├── train_cub200.py         # CUB-200 cross-domain experiments
│   ├── statistical_significance.py
│   └── hyperparameter_interaction.py
├── utils/                       # Utility functions
│   └── model_complexity.py     # FLOPs and parameter analysis
├── visualization/               # Visualization tools
│   └── plot_results.py
├── scripts/                     # Training and evaluation scripts
│   ├── train_teacher.py        # Train ResNet-50 teacher
│   ├── train_distillation.py   # Train student with KD
│   ├── test_model.py           # Model evaluation
│   ├── run_ablation_study.sh   # Run ablation experiments
│   └── compile_paper.sh        # Compile LaTeX paper
├── paper/                       # LaTeX paper source
│   ├── main_en.tex             # English version
│   ├── main_cn.tex             # Chinese version
│   ├── references.bib          # Bibliography
│   └── figures/                # Paper figures
├── docs/                        # Documentation
│   ├── zh/                     # Chinese documentation
│   └── CONTRIBUTING.md
└── templates/                   # LaTeX templates
    └── IEEE_Access/            # IEEE Access template
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/AlexZander-666/orthogonal-learning-food-classification.git
cd orthogonal-learning-food-classification

# Create virtual environment (recommended)
conda create -n orthogonal_learning python=3.8
conda activate orthogonal_learning

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Download Food-101 dataset:

```bash
# Method 1: Automatic download using torchvision
python -c "from torchvision import datasets; datasets.Food101(root='./data', download=True)"

# Method 2: Manual download
# URL: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
# Extract to ./data/food-101/
```

### 3. Train Teacher Model (Optional)

If you have a pre-trained ResNet-50 teacher, skip this step. Otherwise:

```bash
python scripts/train_teacher.py \
    --data-dir ./data \
    --epochs 30 \
    --batch-size 64 \
    --output teacher_resnet50.pth
```

### 4. Train Student Model

#### 4.1 Using ECA Attention + Knowledge Distillation

```bash
python scripts/train_distillation.py \
    --data-dir ./data \
    --attention-type eca \
    --teacher-checkpoint teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.05 \
    --pretrained \
    --output-dir ./checkpoints/eca
```

#### 4.2 Using SimAM Attention + Knowledge Distillation

```bash
python scripts/train_distillation.py \
    --data-dir ./data \
    --attention-type simam \
    --teacher-checkpoint teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.05 \
    --pretrained \
    --output-dir ./checkpoints/simam
```

### 5. Model Complexity Analysis

```bash
python utils/model_complexity.py
```

Expected output:
```
========================================
Model Complexity Analysis
========================================
Total Parameters:    5,483,237 (5.48M)
Trainable:          5,483,237
FLOPs:              219.909M (0.22 G)
Model Size:         20.92 MB
Inference Time:     3.45 ± 0.12 ms
Throughput:         289.86 images/s
========================================
```

### 6. Run Ablation Study

Run complete ablation experiments (test different attention mechanisms and training strategies):

```bash
chmod +x scripts/run_ablation_study.sh
./scripts/run_ablation_study.sh
```

---

## 📈 Experimental Results

### Ablation Study

| Configuration | Attention | Distillation | Accuracy | Parameters |
|--------------|-----------|--------------|----------|------------|
| Baseline | ❌ | ❌ | 74.23% | 5.48M |
| +ECA | ✅ | ❌ | 75.86% | 5.48M |
| +SimAM | ✅ | ❌ | 75.42% | 5.48M |
| +KD Only | ❌ | ✅ | 76.91% | 5.48M |
| **+ECA +KD (Full)** | **✅** | **✅** | **78.50%** | **5.48M** |
| **+SimAM +KD (Full)** | **✅** | **✅** | **78.12%** | **5.48M** |

**Key Finding**: Combined gain (4.27%) ≈ Individual gains (1.63% + 2.68% = 4.31%), demonstrating **orthogonal complementary effects**.

### Attention Mechanism Comparison

| Attention | Parameters | Extra Params | FLOPs | Accuracy | Characteristics |
|-----------|-----------|--------------|-------|----------|-----------------|
| **ECA** | 5.48M | ~500 | 0.22G | **78.50%** | No dimensionality reduction, local interaction |
| **SimAM** | 5.48M | 0 | 0.22G | **78.12%** | Parameter-free, energy function |
| CBAM | 5.52M | 40K | 0.23G | 77.89% | Sequential channel + spatial |
| SE | 5.51M | 30K | 0.22G | 77.65% | Classic channel attention |
| CoordAttention | 5.49M | 10K | 0.23G | 77.92% | Position encoding |

**Insight**: Parameter-efficient designs (ECA, SimAM) outperform heavyweight mechanisms, demonstrating the importance of preserving network capacity for lightweight architectures.

### Cross-Domain Generalization

| Dataset | Baseline | Teacher | Student | Improvement |
|---------|----------|---------|---------|-------------|
| Food-101 | 74.23% | 76.76% | 78.50% | **+4.27%** |
| Flowers-102 | 90.44% | 91.33% | 92.76% | **+2.33%** |

Consistent relative improvements across different visual domains validate the domain-agnostic nature of our approach.

---

## 🔬 Method Overview

### 1. Attention Mechanisms

#### ECA (Efficient Channel Attention)
- **Principle**: Adaptive local cross-channel interaction without dimensionality reduction
- **Advantage**: Minimal parameters (~500), preserves channel information
- **Implementation**: 1D convolution with adaptive kernel size

```python
from models import get_attention_module

eca = get_attention_module('eca', channels=960)
output = eca(input_tensor)
```

#### SimAM (Simple Parameter-Free Attention Module)
- **Principle**: 3D attention based on neuron energy function
- **Advantage**: Zero parameters, plug-and-play
- **Implementation**: Energy-based spatial-channel modeling

```python
simam = get_attention_module('simam')
output = simam(input_tensor)
```

### 2. Knowledge Distillation

Loss function:

```
L = α * L_CE(y, p_student) + (1-α) * T² * KL(p_teacher^T || p_student^T)
```

Where:
- `L_CE`: Hard label cross-entropy loss
- `KL`: KL divergence (soft label loss)
- `T`: Temperature coefficient (default 4.0)
- `α`: Balance coefficient (default 0.7)

### 3. Orthogonal Learning Framework

**Core Hypothesis**: Knowledge distillation and attention mechanisms address two **orthogonal problems** in the learning process:
- **Knowledge Distillation**: Cross-architecture transfer of inductive biases (architectural priors)
- **Attention Mechanisms**: Feature-level attention biases (spatial/channel importance)

This functional orthogonality explains why their performance gains are nearly perfectly additive (4.27% ≈ 4.31%).

---

## 🛠️ Advanced Usage

### Custom Attention Mechanism

Add new attention modules in `models/attention_modules.py`:

```python
class MyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        # Your implementation
        return x

# Register in the dictionary
ATTENTION_MODULES['my_attention'] = MyAttention
```

### Training on Other Datasets

This project supports any ImageFolder format dataset:

```bash
python scripts/train_distillation.py \
    --data-dir /path/to/your/dataset \
    --attention-type eca \
    --epochs 50 \
    --batch-size 32
```

Dataset directory structure:
```
your_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### Export to ONNX

```python
import torch
from models import create_model

model = create_model(num_classes=101, attention_type='eca')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx", 
                  opset_version=11, input_names=['input'], 
                  output_names=['output'])
```

---

## 📝 Citation

If this project helps your research, please cite:

```bibtex
@article{luo2025orthogonal,
  title={Orthogonal Learning: Complementary Biases of Knowledge Distillation and Attention},
  author={Luo, Xiaojuan and Zuo, Bowen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

### Related Papers

- **ECA-Net**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks", CVPR 2020
- **SimAM**: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks", ICML 2021
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv 2015
- **MobileNetV3**: Howard et al., "Searching for MobileNetV3", ICCV 2019

---

## 📄 Paper

The full paper (English and Chinese versions) is available in the `paper/` directory:
- [English Version](paper/main_en.tex)
- [Chinese Version](paper/main_cn.tex)

To compile the paper:
```bash
cd paper
xelatex main_en.tex  # For English version
xelatex main_cn.tex  # For Chinese version
bibtex main_en
xelatex main_en.tex
xelatex main_en.tex
```

Or use the compilation script:
```bash
bash scripts/compile_paper.sh
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent framework
- Thanks to the Food-101 dataset authors
- Thanks to the attention mechanism and knowledge distillation research communities

---

## 📮 Contact

- **Authors**: Xiaojuan Luo, Bowen Zuo
- **Institution**: East China University of Science and Technology
- **Email**: 21011149@mail.ecust.edu.cn
- **GitHub**: https://github.com/AlexZander-666/orthogonal-learning-food-classification

---

## ⭐ Star History

If this project helps you, please give it a star ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=AlexZander-666/orthogonal-learning-food-classification&type=Date)](https://star-history.com/#AlexZander-666/orthogonal-learning-food-classification&Date)

---

## 🌐 Language Versions

- [English README](README.md) (This file)
- [中文文档](docs/zh/README_CN.md)

---

<p align="center">
  Made with ❤️ for efficient deep learning research
</p>
