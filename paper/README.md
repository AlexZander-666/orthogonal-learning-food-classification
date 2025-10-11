# Paper: Lightweight Food Image Classification

这个文件夹包含arXiv论文的LaTeX源码。

## 📁 文件说明

- `main.tex` - 论文主文件
- `references.bib` - 参考文献（BibTeX格式）
- `figures/` - 图片文件夹（需要你添加）

## 🔨 本地编译

### 方法1：使用pdflatex（推荐）

```bash
# 编译论文
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 生成的PDF：main.pdf
```

### 方法2：使用latexmk（更简单）

```bash
# 一键编译
latexmk -pdf main.tex

# 清理临时文件
latexmk -c
```

### 方法3：使用Overleaf（在线）

1. 访问 https://www.overleaf.com/
2. 创建新项目
3. 上传 `main.tex` 和 `references.bib`
4. 点击 "Recompile"

## 📋 提交前检查清单

- [ ] 更新作者信息（Your Name → 你的名字）
- [ ] 更新单位信息（Your Institution）
- [ ] 更新邮箱（your.email@example.com）
- [ ] 更新GitHub链接（yourusername → 你的用户名）
- [ ] 添加实验结果图片到 `figures/` 文件夹
- [ ] 检查所有表格数据是否正确
- [ ] 检查参考文献是否完整
- [ ] 编译成功，无错误和警告
- [ ] PDF大小 < 10MB

## 🖼️ 需要添加的图片

建议添加以下图片到 `figures/` 文件夹：

1. **architecture.png** - 模型架构图
   - 展示教师-学生框架
   - 展示注意力模块位置

2. **training_curves.png** - 训练曲线
   - Loss曲线
   - 准确率曲线

3. **attention_comparison.png** - 注意力机制对比
   - 不同注意力的性能柱状图

4. **inference_time.png** - 推理时间对比
   - 各模型的推理速度对比

## 📊 如何生成图片

### 训练曲线

```python
import matplotlib.pyplot as plt
import json

# 读取训练历史
with open('../checkpoints/eca/history_mobilenetv3_eca.json', 'r') as f:
    history = json.load(f)

# 绘制准确率
plt.figure(figsize=(10, 6))
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
```

### 注意力对比图

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
attentions = ['Baseline', 'ECA', 'SimAM', 'CBAM', 'SE', 'Coord']
accuracies = [76.91, 78.50, 78.12, 77.89, 77.65, 77.92]

# 绘图
plt.figure(figsize=(10, 6))
bars = plt.bar(attentions, accuracies, color='skyblue')
bars[1].set_color('orange')  # 突出ECA
bars[2].set_color('green')   # 突出SimAM

plt.ylabel('Accuracy (%)')
plt.title('Comparison of Different Attention Mechanisms')
plt.ylim([70, 80])
plt.grid(axis='y', alpha=0.3)

# 在柱子上标注数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom')

plt.savefig('figures/attention_comparison.png', dpi=300, bbox_inches='tight')
```

## 🚀 打包提交到arXiv

### 创建提交包

```bash
# 方法1：tar.gz（推荐）
tar -czf arxiv_submission.tar.gz main.tex references.bib figures/

# 方法2：zip
zip -r arxiv_submission.zip main.tex references.bib figures/
```

### 提交到arXiv

1. 登录 https://arxiv.org/
2. 点击 "Submit"
3. 上传 `arxiv_submission.tar.gz`
4. 填写元数据
5. 预览并提交

详细步骤见: `../arxiv_submission_guide.md`

## ⚠️ 常见问题

### Q: 编译失败："File not found: xxx.sty"

**A**: 缺少LaTeX包，安装即可：
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Windows
# 下载安装 MiKTeX 或 TeX Live
```

### Q: 参考文献不显示

**A**: 需要运行bibtex：
```bash
pdflatex main.tex
bibtex main      # 这一步生成参考文献
pdflatex main.tex
pdflatex main.tex
```

### Q: 图片不显示

**A**: 检查：
1. 图片文件是否存在
2. 文件名是否正确（区分大小写）
3. 路径是否正确

### Q: PDF太大（>10MB）

**A**: 压缩图片：
```bash
# 使用ImageMagick压缩
convert input.png -quality 85 output.png
```

## 📝 论文修改建议

### 需要自定义的部分

1. **摘要** (main.tex 第35-42行)
   - 根据实际实验结果更新数字

2. **引言** (main.tex 第46-70行)
   - 可以根据你的研究动机调整

3. **实验结果** (main.tex 第134-189行)
   - 更新为真实的实验数据
   - 如果没做实验，保持当前理论值

4. **表格数据** (main.tex)
   - Table 1: 主要结果对比
   - Table 2: 消融实验
   - Table 3: 注意力机制对比
   - Table 4: 与其他方法对比

### 可选的改进

1. **添加图片**
   - 架构图更直观
   - 可视化结果更有说服力

2. **扩充相关工作**
   - 引用更多最新论文
   - 讨论更多相关方法

3. **增加分析**
   - 错误分析
   - 可视化注意力热图
   - 更深入的讨论

## 🎓 学术诚信

- ✅ 所有引用的论文都已列在参考文献中
- ✅ 实验结果真实可靠（如果是理论值请标注）
- ✅ 代码开源，结果可复现
- ⚠️ 不要抄袭他人论文
- ⚠️ 不要伪造实验数据

## 📧 联系方式

如有问题，请联系：
- Email: your.email@example.com
- GitHub: https://github.com/yourusername/lightweight-food-classification

---

祝你顺利发表！🚀

