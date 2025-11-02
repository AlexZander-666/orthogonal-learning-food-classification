# 贡献指南

感谢你对本项目的关注！我们欢迎任何形式的贡献。

## 🤝 如何贡献

### 报告Bug

如果你发现了Bug，请：

1. 检查是否已有相关Issue
2. 创建新Issue，包含：
   - 清晰的标题
   - 详细的bug描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息（OS、Python版本、PyTorch版本等）
   - 相关代码片段/错误信息

### 提出新功能

如果你有好的想法：

1. 先创建Issue讨论
2. 说明：
   - 功能描述
   - 使用场景
   - 可能的实现方案
3. 等待维护者反馈

### 提交代码

#### 开发流程

1. **Fork仓库**
   ```bash
   # 在GitHub上点击Fork按钮
   git clone https://github.com/你的用户名/orthogonal-learning-food-classification.git
   cd orthogonal-learning-food-classification
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **开发**
   - 遵循代码风格
   - 添加必要的注释
   - 编写/更新测试
   - 更新文档

4. **测试**
   ```bash
   # 运行测试
   python models/attention_modules.py
   python models/mobilenetv3_attention.py
   python utils/model_complexity.py
   ```

5. **提交**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   # 或
   git commit -m "fix: fix critical bug"
   ```

6. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在GitHub上创建Pull Request

#### 提交信息规范

使用语义化提交信息：

- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具相关

示例：
```
feat: add CBAM attention mechanism
fix: correct SimAM energy calculation
docs: update README with new examples
```

#### 代码风格

- 遵循PEP 8
- 使用4空格缩进
- 类名使用CamelCase
- 函数名使用snake_case
- 添加docstring

示例：
```python
class MyAttention(nn.Module):
    """
    My custom attention mechanism.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 实现
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of same shape
        """
        # 实现
        return x
```

### 添加新的注意力机制

如果你想添加新的注意力机制：

1. 在 `models/attention_modules.py` 中添加:
   ```python
   class YourAttention(nn.Module):
       def __init__(self, channels):
           super().__init__()
           # 你的实现
       
       def forward(self, x):
           # 你的实现
           return x
   
   # 注册
   ATTENTION_MODULES['your_attention'] = YourAttention
   ```

2. 添加测试:
   ```python
   if __name__ == "__main__":
       x = torch.randn(2, 64, 32, 32)
       module = YourAttention(64)
       out = module(x)
       print(f"Output shape: {out.shape}")
   ```

3. 更新文档
4. 运行完整测试

### 改进文档

文档改进总是受欢迎的：

- 修正错别字
- 改进示例
- 添加教程
- 翻译文档

## 📋 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/AlexZander-666/orthogonal-learning-food-classification.git
cd orthogonal-learning-food-classification

# 创建虚拟环境
conda create -n food_cls_dev python=3.8
conda activate food_cls_dev

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install pytest flake8 black isort
```

## ✅ PR检查清单

提交PR前确保：

- [ ] 代码遵循项目风格
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了文档
- [ ] 提交信息清晰
- [ ] PR描述完整

## 🎯 优先级

当前欢迎的贡献：

### 高优先级
- [ ] 添加更多注意力机制（Transformer-based等）
- [ ] 支持更多轻量级backbone（EfficientNet, GhostNet等）
- [ ] 优化训练速度
- [ ] 添加可视化工具（Grad-CAM等）

### 中优先级
- [ ] 支持多GPU训练
- [ ] 添加模型量化
- [ ] ONNX/TensorRT部署
- [ ] 更多数据集支持

### 低优先级
- [ ] Web demo
- [ ] 移动端部署示例
- [ ] 更多可视化

## 🌟 贡献者

感谢所有贡献者！

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- 这里会自动生成贡献者列表 -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 📞 联系方式

如有问题：

- 创建Issue: https://github.com/AlexZander-666/orthogonal-learning-food-classification/issues
- 邮件: 21011149@mail.ecust.edu.cn

## 📄 许可证

贡献的代码将采用MIT许可证。

---

再次感谢你的贡献！🎉


