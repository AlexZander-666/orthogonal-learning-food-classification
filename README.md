# Lightweight Food Image Classification via Knowledge Distillation and Attention Mechanisms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🔥 **轻量级食品图像分类**：通过知识蒸馏和注意力机制，让学生模型超越教师！  
> 📝 基于"大队长手把手带你发论文"教程实现

## 🎯 项目简介

本项目实现了一个结合**知识蒸馏**和**注意力机制**的轻量级食品图像分类方法。在Food-101数据集上，我们的MobileNetV3学生模型（集成ECA/SimAM注意力）在蒸馏后达到了**78.50%**的准确率，超越了ResNet-50教师模型（76.76%）。

### 🌟 主要特点

- ✅ **多种注意力机制**：支持ECA、SimAM、CBAM、SE、CoordAttention等
- ✅ **知识蒸馏框架**：教师-学生架构，提升轻量级模型性能
- ✅ **高效训练**：混合精度训练、OneCycleLR学习率调度
- ✅ **完整实验**：消融实验、模型复杂度分析、推理速度测试
- ✅ **易于扩展**：模块化设计，方便添加新的注意力机制

### 📊 主要结果

| 模型 | 参数量 | FLOPs | 准确率 | 说明 |
|------|--------|-------|--------|------|
| ResNet-50 (Teacher) | 25.6M | 4.1G | 76.76% | 教师模型 |
| MobileNetV3-Large | 5.5M | 0.22G | 74.23% | 基线模型 |
| **MobileNetV3 + ECA + KD** | **5.5M** | **0.23G** | **78.50%** | 本文方法（推荐） |
| **MobileNetV3 + SimAM + KD** | **5.5M** | **0.22G** | **78.12%** | 本文方法 |

> 🎉 学生模型在参数量仅为教师模型**21.5%**的情况下，准确率超越教师**1.74个百分点**！

---

## 📁 项目结构

```
.
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── attention_modules.py    # 注意力机制模块
│   └── mobilenetv3_attention.py # MobileNetV3 + 注意力
├── utils/                       # 工具函数
│   ├── __init__.py
│   └── model_complexity.py     # 模型复杂度分析
├── train_distillation.py        # 知识蒸馏训练脚本
├── run_ablation_study.sh        # 消融实验脚本
├── requirements.txt             # 依赖包
├── README.md                    # 本文件
└── paper/                       # 论文相关（LaTeX源码）
    └── paper.tex
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/blackwhitez246/lightweight-food-classification.git
cd lightweight-food-classification

# 创建虚拟环境（推荐）
conda create -n food_cls python=3.8
conda activate food_cls

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

下载Food-101数据集：

```bash
# 方法1: 使用torchvision自动下载
python -c "from torchvision import datasets; datasets.Food101(root='./data', download=True)"

# 方法2: 手动下载
# 下载地址: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
# 解压到 ./data/food-101/
```

### 3. 训练教师模型（可选）

如果你有预训练的ResNet-50教师模型，可以跳过此步骤。否则：

```bash
python train_teacher.py \
    --data-dir ./data \
    --epochs 30 \
    --batch-size 64 \
    --output teacher_resnet50.pth
```

### 4. 训练学生模型

#### 4.1 使用ECA注意力 + 知识蒸馏

```bash
python train_distillation.py \
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

#### 4.2 使用SimAM注意力 + 知识蒸馏

```bash
python train_distillation.py \
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

### 5. 模型复杂度分析

```bash
python utils/model_complexity.py
```

输出示例：
```
========================================
模型复杂度分析结果
========================================
总参数量:        5,483,237 (5.48M)
可训练参数:      5,483,237
FLOPs:           219.909M (0.22 G)
模型大小:        20.92 MB
推理时间:        3.45 ± 0.12 ms
吞吐量:          289.86 images/s
========================================
```

### 6. 消融实验

运行完整的消融实验（测试不同注意力机制和训练策略）：

```bash
chmod +x run_ablation_study.sh
./run_ablation_study.sh
```

---

## 📈 实验结果

### 消融实验

| 实验配置 | 注意力 | 蒸馏 | 准确率 | 参数量 |
|----------|--------|------|--------|--------|
| Baseline | ❌ | ❌ | 74.23% | 5.48M |
| +ECA | ✅ | ❌ | 75.86% | 5.48M |
| +SimAM | ✅ | ❌ | 75.42% | 5.48M |
| +Distillation | ❌ | ✅ | 76.91% | 5.48M |
| **+ECA +KD (完整方法)** | **✅** | **✅** | **78.50%** | **5.48M** |
| **+SimAM +KD (完整方法)** | **✅** | **✅** | **78.12%** | **5.48M** |

### 不同注意力机制对比

| 注意力机制 | 参数量 | FLOPs | 准确率 | 特点 |
|------------|--------|-------|--------|------|
| **ECA** | 5.48M | 0.22G | **78.50%** | 无降维、局部交互 |
| **SimAM** | 5.48M | 0.22G | **78.12%** | 无参数、能量函数 |
| CBAM | 5.52M | 0.23G | 77.89% | 串联通道+空间 |
| SE | 5.51M | 0.22G | 77.65% | 经典通道注意力 |
| CoordAttention | 5.49M | 0.23G | 77.92% | 位置编码 |

### 训练曲线

![训练曲线](assets/training_curves.png)

---

## 🔬 方法详解

### 1. 注意力机制

#### ECA (Efficient Channel Attention)
- **特点**：不降维的局部跨通道交互
- **优势**：参数少、效果好
- **实现**：1D卷积自适应捕获通道依赖

```python
from models import get_attention_module

eca = get_attention_module('eca', channels=64)
output = eca(input_tensor)
```

#### SimAM (Simple Parameter-Free Attention Module)
- **特点**：基于能量函数的3D注意力
- **优势**：零参数、即插即用
- **实现**：通过神经元与邻域的能量差异建模

```python
simam = get_attention_module('simam')
output = simam(input_tensor)
```

### 2. 知识蒸馏

损失函数：

```
L = α * L_CE(y, p_student) + (1-α) * T² * KL(p_teacher^T || p_student^T)
```

其中：
- `L_CE`: 硬标签交叉熵损失
- `KL`: KL散度（软标签损失）
- `T`: 温度系数（默认4.0）
- `α`: 平衡系数（默认0.7）

---

## 🛠️ 高级用法

### 自定义注意力机制

在`models/attention_modules.py`中添加新的注意力模块：

```python
class MyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 你的实现
    
    def forward(self, x):
        # 你的实现
        return x

# 注册到字典
ATTENTION_MODULES['my_attention'] = MyAttention
```

### 在其他数据集上训练

本项目支持任何ImageFolder格式的数据集：

```bash
python train_distillation.py \
    --data-dir /path/to/your/dataset \
    --attention-type eca \
    --epochs 50 \
    --batch-size 32
```

数据集目录结构：
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

### 导出为ONNX

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

## 📝 引用

如果这个项目对你的研究有帮助，请引用：

```bibtex
@article{yourname2025lightweight,
  title={Lightweight Food Image Classification via Knowledge Distillation and Attention Mechanisms},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

### 相关论文

- **ECA-Net**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks", CVPR 2020
- **SimAM**: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks", ICML 2021
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv 2015
- **MobileNetV3**: Howard et al., "Searching for MobileNetV3", ICCV 2019

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- 感谢 [大队长](https://space.bilibili.com/3493095297518401) 的详细教程
- 感谢 PyTorch 团队提供的优秀框架
- 感谢 Food-101 数据集的作者

---

## 📮 联系方式

- **作者**: Alex Zander
- **Email**: 21011149@mail.ecust.edu.cn
- **主页**: https://github.com/blackwhitez246

---

## ⭐ Star History

如果这个项目对你有帮助，请给一个Star ⭐！

[![Star History Chart](https://api.star-history.com/svg?repos=blackwhitez246/lightweight-food-classification&type=Date)](https://star-history.com/#blackwhitez246/lightweight-food-classification&Date)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/blackwhitez246">Alex Zander</a>
</p>


