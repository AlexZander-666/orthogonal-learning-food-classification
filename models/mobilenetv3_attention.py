"""
MobileNetV3 with Attention Mechanisms
支持多种注意力机制的MobileNetV3模型
"""

import torch
import torch.nn as nn
from torchvision import models
from .attention_modules import get_attention_module


class MobileNetV3_Attention(nn.Module):
    """
    MobileNetV3 + 注意力机制模型
    
    Args:
        num_classes: 分类类别数
        attention_type: 注意力机制类型 ('eca', 'simam', 'cbam', 'se', 'coord', 'none')
        pretrained: 是否使用预训练权重
        model_size: 'large' 或 'small'
    """
    def __init__(self, num_classes=101, attention_type='eca', pretrained=False, model_size='large'):
        super(MobileNetV3_Attention, self).__init__()
        
        self.attention_type = attention_type
        
        # 加载 MobileNetV3
        if model_size == 'large':
            if pretrained:
                mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            else:
                mobilenet = models.mobilenet_v3_large(weights=None)
        else:
            if pretrained:
                mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                mobilenet = models.mobilenet_v3_small(weights=None)
        
        self.features = mobilenet.features
        
        # 获取最后一层的输出通道数
        last_channel = mobilenet.classifier[0].in_features
        
        # 添加注意力模块
        if attention_type.lower() != 'none':
            self.attention = get_attention_module(attention_type, last_channel)
        else:
            self.attention = nn.Identity()
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        
        # 注意力机制
        x = self.attention(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        attention_params = sum(p.numel() for p in self.attention.parameters())
        
        info = {
            'attention_type': self.attention_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'attention_params': attention_params,
            'attention_ratio': f'{attention_params / total_params * 100:.2f}%'
        }
        
        return info


def create_model(num_classes=101, attention_type='eca', pretrained=False, model_size='large'):
    """
    创建模型的便捷函数
    
    Args:
        num_classes: 分类类别数
        attention_type: 注意力机制类型
        pretrained: 是否使用预训练
        model_size: 模型大小
    
    Returns:
        模型实例
    """
    model = MobileNetV3_Attention(
        num_classes=num_classes,
        attention_type=attention_type,
        pretrained=pretrained,
        model_size=model_size
    )
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("MobileNetV3 + Attention 模型测试")
    print("=" * 70)
    
    attention_types = ['eca', 'simam', 'cbam', 'se', 'coord', 'none']
    
    x = torch.randn(2, 3, 224, 224)
    print(f"\n输入张量形状: {x.shape}")
    
    for att_type in attention_types:
        print(f"\n{'='*70}")
        print(f"测试 MobileNetV3-Large + {att_type.upper()}")
        print(f"{'='*70}")
        
        model = create_model(num_classes=101, attention_type=att_type, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        info = model.get_model_info()
        
        print(f"✓ 输出形状: {out.shape}")
        print(f"✓ 总参数量: {info['total_params']:,}")
        print(f"✓ 可训练参数: {info['trainable_params']:,}")
        print(f"✓ 注意力模块参数: {info['attention_params']:,} ({info['attention_ratio']})")
    
    print(f"\n{'='*70}")
    print("所有模型测试完成！")
    print(f"{'='*70}\n")


