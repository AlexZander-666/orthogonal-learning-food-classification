# Paper: Orthogonal Learning - Complementary Biases of Knowledge Distillation and Attention

This directory contains the LaTeX source files for the research paper.

## 📄 Available Versions

- **`main_en.tex`**: English version of the paper
- **`main_cn.tex`**: Chinese version of the paper (中文版本)
- **`references.bib`**: Shared bibliography file

## 🖼️ Figures

All paper figures are located in the `figures/` subdirectory:
- `ablation_study.png`: Ablation study visualization
- `attention_comparison.png`: Attention mechanism comparison
- `accuracy_params_tradeoff.png`: Accuracy vs parameters trade-off
- `cross_dataset_comparison.png`: Cross-dataset generalization results
- `flowers102_training.png`: Training curves on Flowers-102
- `food101_training_curves.png`: Training curves on Food-101

## 🔨 Compilation

### Prerequisites

Make sure you have a LaTeX distribution installed:
- **Windows**: MiKTeX or TeX Live
- **macOS**: MacTeX
- **Linux**: TeX Live

Required packages:
- For English version: `article`, standard LaTeX packages
- For Chinese version: `ctexart`, XeLaTeX support

### Compile English Version

```bash
cd paper
xelatex main_en.tex
bibtex main_en
xelatex main_en.tex
xelatex main_en.tex
```

Or use `pdflatex` if XeLaTeX is not available:
```bash
pdflatex main_en.tex
bibtex main_en
pdflatex main_en.tex
pdflatex main_en.tex
```

### Compile Chinese Version

**Note**: Chinese version requires XeLaTeX for proper font rendering.

```bash
cd paper
xelatex main_cn.tex
bibtex main_cn
xelatex main_cn.tex
xelatex main_cn.tex
```

### Quick Compilation Script

Use the provided compilation scripts:

**Linux/macOS:**
```bash
bash ../scripts/compile_paper.sh
```

**Windows:**
```cmd
..\scripts\compile_paper.bat
```

## 📋 Paper Structure

### English Version (main_en.tex)

1. **Abstract**: Overview of the lightweight food classification method
2. **Introduction**: Problem motivation and main contributions
3. **Related Work**: Knowledge distillation, lightweight networks, attention mechanisms
4. **Method**: Overall framework, attention mechanisms, knowledge distillation
5. **Experiments**: Setup, main results, ablation studies, comparisons
6. **Conclusion**: Summary and future work

### Chinese Version (main_cn.tex)

中文版包含与英文版相同的内容结构，并增加了以下部分：
- 正交学习理论框架的详细阐述
- 更详细的实验分析和讨论
- 方法论洞见和实践指导

## 🔧 Troubleshooting

### Missing Packages

If you encounter missing package errors:

**TeX Live/MacTeX:**
```bash
tlmgr install <package-name>
```

**MiKTeX:**
- Open MiKTeX Console
- Go to "Packages" tab
- Search and install missing packages

### Chinese Font Issues

If Chinese characters don't display correctly:

1. Make sure you're using XeLaTeX (not pdflatex)
2. Install Chinese fonts on your system
3. Update the font settings in `main_cn.tex` if needed

### Figure Not Found Errors

Ensure all figure files are in the `figures/` subdirectory. The LaTeX files use relative paths like:
```latex
\includegraphics[width=\columnwidth]{figures/ablation_study.png}
```

## 📊 Generating Figures

If you need to regenerate the paper figures from experimental results:

```bash
python ../visualization/plot_results.py
```

This will create all necessary PNG files in the `figures/` directory.

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{luo2025orthogonal,
  title={Orthogonal Learning: Complementary Biases of Knowledge Distillation and Attention},
  author={Luo, Xiaojuan and Zuo, Bowen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📧 Contact

For questions about the paper or LaTeX source:
- **Email**: 21011149@mail.ecust.edu.cn
- **GitHub Issues**: https://github.com/AlexZander-666/orthogonal-learning-food-classification/issues

## 📚 Additional Resources

- [Overleaf LaTeX Tutorial](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [中文LaTeX指南](https://liam.page/2014/09/08/latex-introduction/)
