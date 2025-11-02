# 论文编译指南

## 📄 文件说明

### 论文版本

- **`main_cn.tex`** - 中文版（华理学报格式，双栏）⭐ **推荐用于国内投稿**
- **`main.tex`** - 英文版（国际期刊格式，单栏）⭐ **推荐用于arXiv/国际期刊**

### 编译脚本

- **`compile_cn.bat`** - Windows编译脚本（中文版）
- **`compile_cn.sh`** - Linux/Mac编译脚本（中文版）

---

## 🚀 快速开始

### 编译中文版

**Windows**：
```batch
compile_cn.bat
```

**Linux/Mac**：
```bash
chmod +x compile_cn.sh
./compile_cn.sh
```

**手动编译**：
```bash
xelatex main_cn.tex
bibtex main_cn
xelatex main_cn.tex
xelatex main_cn.tex
```

### 编译英文版

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📋 编译要求

### 中文版（main_cn.tex）
- **编译器**：XeLaTeX（必需，支持中文）
- **宏包**：ctex, geometry, booktabs, amsmath 等
- **字体**：系统中文字体（宋体、黑体、楷体）

### 英文版（main.tex）
- **编译器**：PDFLaTeX 或 XeLaTeX
- **宏包**：geometry, graphicx, amsmath, booktabs 等

---

## 🎯 作者信息

### 中文版
- **一作**：罗小娟（副教授）
- **二作**：左博文
- **单位**：华东理工大学 信息科学与工程学院
- **地址**：上海 200237

### 英文版
- **作者**：Alex Zander
- **单位**：East China University of Science and Technology
- **邮箱**：21011149@mail.ecust.edu.cn

---

## 📊 关键实验结果

### Food-101数据集
- 基线：74.23%
- 教师：76.76%
- **学生（ECA+KD）**：**78.50%** (+4.27%)
- **学生（SimAM+KD）**：**78.12%** (+3.89%)

### Oxford Flowers-102数据集
- 基线：90.44%
- 教师：91.33%
- **学生（SimAM+KD）**：**92.76%** (+2.33%)

---

## ⚠️ 常见问题

### Q1: 编译时提示找不到ctex宏包
**解决**：
```bash
# TeX Live
tlmgr install ctex

# MiKTeX（自动安装）
# 首次编译会自动下载
```

### Q2: 中文显示为方框或乱码
**解决**：
- 确保使用XeLaTeX编译（不是PDFLaTeX）
- 检查系统是否安装中文字体

### Q3: 参考文献不显示
**解决**：
- 按完整顺序编译4次
- 检查.bib文件是否在同一目录

### Q4: 公式或表格错位
**解决**：
- 检查是否有未闭合的括号
- 查看.log文件中的错误信息

---

## 📚 参考文献

参考文献在 `references.bib` 中管理，包含19篇文献：
- 基础方法论文（Hinton, He, Howard等）
- 注意力机制论文（Wang, Yang, Hou等）
- 最新KD进展（Zhou, Furlanello, Ji等）

---

## 🔗 相关资源

- **代码仓库**：https://github.com/blackwhitez246/lightweight-food-classification
- **Food-101数据集**：https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **Flowers-102数据集**：torchvision自动下载
- **实验脚本**：`../experiments/` 目录

---

## ✉️ 联系方式

如有问题，请联系：
- **邮箱**：21011149@mail.ecust.edu.cn
- **GitHub**：https://github.com/blackwhitez246

---

**生成日期**：2025年11月2日  
**状态**：✅ 准备就绪，可投稿




