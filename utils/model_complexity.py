"""
模型复杂度分析工具
计算参数量、FLOPs、推理速度等指标
"""

import torch
import time
import numpy as np
from thop import profile, clever_format


def count_parameters(model):
    """
    计算模型参数量
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    计算FLOPs(浮点运算次数)
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
    
    Returns:
        flops: FLOPs
        params: 参数量
    """
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(input_size).to(device)
    
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, params, flops_str, params_str
    except Exception as e:
        print(f"[WARN] 计算FLOPs失败: {e}")
        return None, None, None, None


def measure_inference_time(model, input_size=(1, 3, 224, 224), 
                          device='cuda', num_runs=100, warmup=10):
    """
    测量推理时间
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
        num_runs: 测试次数
        warmup: 预热次数
    
    Returns:
        avg_time: 平均推理时间(ms)
        std_time: 标准差
        throughput: 吞吐量(images/s)
    """
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # 同步CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测量时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1000 / avg_time * input_size[0]  # images/s
    
    return avg_time, std_time, throughput


def get_model_size(model):
    """
    计算模型文件大小(MB)
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    return size_mb


def analyze_model(model, input_size=(1, 3, 224, 224), device='cuda', verbose=True):
    """
    全面分析模型复杂度
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
        verbose: 是否打印结果
    
    Returns:
        results: 结果字典
    """
    results = {}
    
    # 1. 参数量
    total_params, trainable_params = count_parameters(model)
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    results['total_params_M'] = total_params / 1e6
    
    # 2. FLOPs
    flops, params, flops_str, params_str = compute_flops(model, input_size, 'cpu')
    results['flops'] = flops
    results['flops_str'] = flops_str if flops_str else 'N/A'
    results['flops_G'] = flops / 1e9 if flops else None
    
    # 3. 模型大小
    model_size = get_model_size(model)
    results['model_size_mb'] = model_size
    
    # 4. 推理时间
    if torch.cuda.is_available() and device == 'cuda':
        try:
            avg_time, std_time, throughput = measure_inference_time(
                model, input_size, device
            )
            results['inference_time_ms'] = avg_time
            results['inference_std_ms'] = std_time
            results['throughput_imgs_per_sec'] = throughput
        except Exception as e:
            print(f"[WARN] 测量推理时间失败: {e}")
            results['inference_time_ms'] = None
            results['throughput_imgs_per_sec'] = None
    
    # 打印结果
    if verbose:
        print("\n" + "="*70)
        print("模型复杂度分析结果")
        print("="*70)
        print(f"总参数量:        {results['total_params']:,} ({results['total_params_M']:.2f}M)")
        print(f"可训练参数:      {results['trainable_params']:,}")
        print(f"FLOPs:           {results['flops_str']} ({results.get('flops_G', 'N/A')} G)")
        print(f"模型大小:        {results['model_size_mb']:.2f} MB")
        
        if results.get('inference_time_ms'):
            print(f"推理时间:        {results['inference_time_ms']:.2f} ± {results['inference_std_ms']:.2f} ms")
            print(f"吞吐量:          {results['throughput_imgs_per_sec']:.2f} images/s")
        print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    from models import create_model
    
    print("="*70)
    print("模型复杂度分析工具测试")
    print("="*70)
    
    attention_types = ['eca', 'simam', 'cbam', 'none']
    
    all_results = {}
    
    for att_type in attention_types:
        print(f"\n分析 MobileNetV3-Large + {att_type.upper()}")
        
        model = create_model(
            num_classes=101,
            attention_type=att_type,
            pretrained=False,
            model_size='large'
        )
        
        results = analyze_model(model, verbose=True)
        all_results[att_type] = results
    
    # 对比表格
    print("\n" + "="*70)
    print("对比表格")
    print("="*70)
    print(f"{'Attention':<12} {'Params(M)':<12} {'FLOPs':<15} {'Size(MB)':<12}")
    print("-"*70)
    
    for att_type, results in all_results.items():
        print(f"{att_type.upper():<12} "
              f"{results['total_params_M']:<12.2f} "
              f"{results['flops_str']:<15} "
              f"{results['model_size_mb']:<12.2f}")
    
    print("="*70)


