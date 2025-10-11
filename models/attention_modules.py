"""
注意力机制模块集合
包含 ECA-Net, SimAM, CBAM 等多种注意力机制
参考: 大队长手把手带你发论文第三版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """
    ECA-Net: Efficient Channel Attention (CVPR 2020)
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    特点: 不降维的局部跨通道交互策略，参数少效果好
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 自适应确定1D卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        
        # 1D卷积: (B, C, 1, 1) -> (B, 1, C) -> (B, 1, C) -> (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # 生成通道注意力权重
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module (MM 2024)
    论文: A Lightweight Multi-domain Multi-attention Progressive Network
    特点: 无参数、基于能量函数的3D注意力机制
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        
        # 计算空间维度上的均值
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        
        # 计算 (x - mean)^2
        x_minus_mu_square = (x - x_mean).pow(2)
        
        # 计算能量函数的倒数（显著性）
        # E = 4 * (σ² + λ) / [(t-μ)² / (σ²+λ) + 2]
        # 1/E ≈ (t-μ)² / [4(σ²+λ)] + 0.5
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        y = x_minus_mu_square / (4 * (variance + self.e_lambda)) + 0.5
        
        # sigmoid 激活并与原始输入相乘
        return x * y.sigmoid()


class CBAM_ChannelAttention(nn.Module):
    """
    CBAM通道注意力模块
    """
    def __init__(self, channels, reduction=16):
        super(CBAM_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class CBAM_SpatialAttention(nn.Module):
    """
    CBAM空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super(CBAM_SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module (ECCV 2018)
    论文: CBAM: Convolutional Block Attention Module
    特点: 串联通道注意力和空间注意力
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = CBAM_ChannelAttention(channels, reduction)
        self.spatial_attention = CBAM_SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    SE-Net: Squeeze-and-Excitation Networks (CVPR 2018)
    论文: Squeeze-and-Excitation Networks
    特点: 通过全局池化和全连接层建模通道依赖性
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    """
    Coordinate Attention (CVPR 2021)
    论文: Coordinate Attention for Efficient Mobile Network Design
    特点: 将位置信息嵌入到通道注意力中
    """
    def __init__(self, channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        return out


# 注意力机制字典，方便选择
ATTENTION_MODULES = {
    'eca': ECA,
    'simam': SimAM,
    'cbam': CBAM,
    'se': SEBlock,
    'coord': CoordAttention,
}


def get_attention_module(name, channels, **kwargs):
    """
    根据名称获取注意力模块
    
    Args:
        name: 注意力机制名称 ('eca', 'simam', 'cbam', 'se', 'coord')
        channels: 输入通道数
        **kwargs: 其他参数
    
    Returns:
        注意力模块实例
    """
    if name.lower() not in ATTENTION_MODULES:
        raise ValueError(f"未知的注意力机制: {name}. 可选: {list(ATTENTION_MODULES.keys())}")
    
    attention_class = ATTENTION_MODULES[name.lower()]
    
    # SimAM不需要channels参数
    if name.lower() == 'simam':
        return attention_class(**kwargs)
    else:
        return attention_class(channels, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("注意力机制模块测试")
    print("=" * 60)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"\n输入张量形状: {x.shape}")
    
    for name in ATTENTION_MODULES.keys():
        print(f"\n测试 {name.upper()}:")
        module = get_attention_module(name, 64)
        out = module(x)
        print(f"  输出形状: {out.shape}")
        print(f"  参数量: {sum(p.numel() for p in module.parameters()):,}")
        print(f"  ✓ 测试通过")
    
    print("\n" + "=" * 60)
    print("所有注意力模块测试完成！")
    print("=" * 60)


