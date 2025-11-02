#!/usr/bin/env python3
"""
超参数交互分析脚本
分析知识蒸馏中温度T和权重α的交互作用
生成2D热力图展示最优组合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
from itertools import product

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3_attention import MobileNetV3WithAttention


def get_data_loaders(data_dir='./data', batch_size=64, subset_ratio=0.2):
    """
    Create data loaders with optional subset for faster experiments
    
    Args:
        data_dir: Path to dataset
        batch_size: Batch size
        subset_ratio: Ratio of data to use (1.0 for full dataset)
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.Food101(
        root=data_dir,
        split='train',
        transform=transform_train,
        download=False
    )
    
    val_dataset = datasets.Food101(
        root=data_dir,
        split='test',
        transform=transform_val,
        download=False
    )
    
    # Create subset if needed
    if subset_ratio < 1.0:
        train_size = int(len(train_dataset) * subset_ratio)
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        
        val_size = int(len(val_dataset) * subset_ratio)
        val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_with_hyperparams(
    temperature,
    alpha,
    teacher,
    train_loader,
    val_loader,
    attention_type='simam',
    epochs=20,
    lr=0.05,
    device='cuda'
):
    """
    Train student model with specific hyperparameters
    
    Args:
        temperature: Distillation temperature
        alpha: Weight for hard label loss
        teacher: Teacher model
        train_loader: Training data loader
        val_loader: Validation data loader
        attention_type: Attention mechanism type
        epochs: Training epochs
        lr: Learning rate
        device: Device to use
        
    Returns:
        Best validation accuracy
    """
    # Create student model
    student = MobileNetV3WithAttention(
        num_classes=101,
        attention_type=attention_type,
        pretrained=True
    )
    student = student.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        student.train()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            student_logits = student(images)
            
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            # Compute losses
            loss_ce = criterion_ce(student_logits, labels)
            loss_kd = criterion_kd(
                nn.functional.log_softmax(student_logits / temperature, dim=1),
                nn.functional.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature ** 2)
            
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validate
        student.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    return best_acc


def hyperparameter_grid_search(
    teacher_checkpoint,
    data_dir='./data',
    output_dir='./experiments/hyperparameter_results',
    temperature_range=None,
    alpha_range=None,
    attention_type='simam',
    epochs=20,
    subset_ratio=0.2,
    device='cuda'
):
    """
    Perform grid search over temperature and alpha hyperparameters
    
    Args:
        teacher_checkpoint: Path to teacher model checkpoint
        data_dir: Path to dataset
        output_dir: Directory to save results
        temperature_range: List of temperature values to try
        alpha_range: List of alpha values to try
        attention_type: Attention mechanism type
        epochs: Training epochs per configuration
        subset_ratio: Ratio of data to use (for faster search)
        device: Device to use
    """
    print(f"\n{'='*80}")
    print("Hyperparameter Grid Search: Temperature × Alpha")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Default ranges
    if temperature_range is None:
        temperature_range = [2.0, 3.0, 4.0, 5.0, 6.0]
    if alpha_range is None:
        alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"Temperature range: {temperature_range}")
    print(f"Alpha range: {alpha_range}")
    print(f"Total configurations: {len(temperature_range) * len(alpha_range)}")
    print(f"Epochs per configuration: {epochs}")
    print(f"Dataset subset ratio: {subset_ratio}\n")
    
    # Load teacher model
    print("Loading teacher model...")
    teacher = models.resnet50()
    teacher.fc = nn.Linear(teacher.fc.in_features, 101)
    checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher = teacher.to(device)
    teacher.eval()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        subset_ratio=subset_ratio
    )
    
    # Grid search
    results = np.zeros((len(alpha_range), len(temperature_range)))
    
    total_configs = len(temperature_range) * len(alpha_range)
    pbar = tqdm(
        product(enumerate(alpha_range), enumerate(temperature_range)),
        total=total_configs,
        desc="Grid Search"
    )
    
    for (alpha_idx, alpha), (temp_idx, temperature) in pbar:
        pbar.set_description(f"T={temperature:.1f}, α={alpha:.1f}")
        
        # Train with these hyperparameters
        acc = train_with_hyperparams(
            temperature=temperature,
            alpha=alpha,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            attention_type=attention_type,
            epochs=epochs,
            device=device
        )
        
        results[alpha_idx, temp_idx] = acc
        pbar.set_postfix({'acc': f'{acc:.2f}%'})
    
    # Find best configuration
    best_idx = np.unravel_index(results.argmax(), results.shape)
    best_alpha = alpha_range[best_idx[0]]
    best_temp = temperature_range[best_idx[1]]
    best_acc = results[best_idx]
    
    # Print summary
    print(f"\n{'='*80}")
    print("Grid Search Results")
    print(f"{'='*80}")
    print(f"Best configuration:")
    print(f"  Temperature: {best_temp:.1f}")
    print(f"  Alpha: {best_alpha:.1f}")
    print(f"  Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # Save results
    results_dict = {
        'temperature_range': temperature_range,
        'alpha_range': alpha_range,
        'results': results.tolist(),
        'best_temperature': float(best_temp),
        'best_alpha': float(best_alpha),
        'best_accuracy': float(best_acc)
    }
    
    with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Visualize results
    visualize_hyperparameter_heatmap(
        results,
        temperature_range,
        alpha_range,
        output_dir
    )
    
    # Create detailed analysis plots
    create_analysis_plots(
        results,
        temperature_range,
        alpha_range,
        output_dir
    )
    
    return results_dict


def visualize_hyperparameter_heatmap(
    results,
    temperature_range,
    alpha_range,
    output_dir
):
    """Create heatmap visualization of hyperparameter search results"""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        results,
        xticklabels=[f'{t:.1f}' for t in temperature_range],
        yticklabels=[f'{a:.1f}' for a in alpha_range],
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Validation Accuracy (%)'},
        vmin=results.min(),
        vmax=results.max(),
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel('Temperature (T)', fontsize=14, fontweight='bold')
    plt.ylabel('Alpha (α)', fontsize=14, fontweight='bold')
    plt.title('Hyperparameter Interaction: Temperature × Alpha\nValidation Accuracy (%)',
              fontsize=16, fontweight='bold', pad=20)
    
    # Mark best configuration
    best_idx = np.unravel_index(results.argmax(), results.shape)
    plt.scatter(best_idx[1] + 0.5, best_idx[0] + 0.5,
               marker='*', s=500, c='blue', edgecolors='white', linewidths=2,
               label=f'Best: T={temperature_range[best_idx[1]]:.1f}, α={alpha_range[best_idx[0]]:.1f}')
    
    plt.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0, -0.1))
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'hyperparameter_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_file}")


def create_analysis_plots(results, temperature_range, alpha_range, output_dir):
    """Create additional analysis plots"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Temperature for different Alpha values
    for i, alpha in enumerate(alpha_range):
        axes[0].plot(temperature_range, results[i, :],
                    marker='o', linewidth=2, label=f'α={alpha:.1f}')
    
    axes[0].set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy vs Temperature for Different α Values',
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Alpha for different Temperature values
    for j, temp in enumerate(temperature_range):
        axes[1].plot(alpha_range, results[:, j],
                    marker='s', linewidth=2, label=f'T={temp:.1f}')
    
    axes[1].set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy vs Alpha for Different T Values',
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'hyperparameter_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plots saved to {output_file}")
    
    # Create 3D surface plot
    create_3d_surface_plot(results, temperature_range, alpha_range, output_dir)


def create_3d_surface_plot(results, temperature_range, alpha_range, output_dir):
    """Create 3D surface plot of hyperparameter interaction"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    T, A = np.meshgrid(temperature_range, alpha_range)
    
    # Plot surface
    surf = ax.plot_surface(T, A, results, cmap='viridis',
                          alpha=0.8, edgecolor='none')
    
    # Add contour lines
    ax.contour(T, A, results, zdir='z', offset=results.min() - 1,
              cmap='viridis', alpha=0.5, linewidths=1)
    
    ax.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Validation Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('3D Surface: Hyperparameter Interaction\nTemperature × Alpha',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Accuracy (%)')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'hyperparameter_3d_surface.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3D surface plot saved to {output_file}")


def analyze_sensitivity(results, temperature_range, alpha_range, output_dir):
    """Analyze sensitivity to each hyperparameter"""
    print(f"\n{'='*80}")
    print("Sensitivity Analysis")
    print(f"{'='*80}\n")
    
    # Temperature sensitivity (variance across temperatures for each alpha)
    temp_sensitivity = np.var(results, axis=1)
    
    # Alpha sensitivity (variance across alphas for each temperature)
    alpha_sensitivity = np.var(results, axis=0)
    
    print("Temperature Sensitivity (Variance across T for each α):")
    for i, alpha in enumerate(alpha_range):
        print(f"  α={alpha:.1f}: {temp_sensitivity[i]:.4f}")
    print(f"  Average: {temp_sensitivity.mean():.4f}\n")
    
    print("Alpha Sensitivity (Variance across α for each T):")
    for j, temp in enumerate(temperature_range):
        print(f"  T={temp:.1f}: {alpha_sensitivity[j]:.4f}")
    print(f"  Average: {alpha_sensitivity.mean():.4f}\n")
    
    # Overall sensitivity
    overall_variance = np.var(results)
    print(f"Overall variance: {overall_variance:.4f}")
    print(f"Range: [{results.min():.2f}%, {results.max():.2f}%]")
    print(f"Span: {results.max() - results.min():.2f}%\n")
    
    # Save sensitivity analysis
    sensitivity = {
        'temperature_sensitivity': {
            f'alpha_{a:.1f}': float(temp_sensitivity[i])
            for i, a in enumerate(alpha_range)
        },
        'alpha_sensitivity': {
            f'temperature_{t:.1f}': float(alpha_sensitivity[j])
            for j, t in enumerate(temperature_range)
        },
        'overall_variance': float(overall_variance),
        'accuracy_range': [float(results.min()), float(results.max())],
        'accuracy_span': float(results.max() - results.min())
    }
    
    with open(os.path.join(output_dir, 'sensitivity_analysis.json'), 'w') as f:
        json.dump(sensitivity, f, indent=2)
    
    print(f"Sensitivity analysis saved to {os.path.join(output_dir, 'sensitivity_analysis.json')}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter interaction analysis for knowledge distillation'
    )
    parser.add_argument('--teacher-checkpoint', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to Food-101 dataset')
    parser.add_argument('--output-dir', type=str, default='./experiments/hyperparameter_results',
                       help='Directory to save results')
    parser.add_argument('--attention-type', type=str, default='simam',
                       choices=['eca', 'simam', 'cbam', 'se', 'coordatt'],
                       help='Attention mechanism type')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs per configuration')
    parser.add_argument('--subset-ratio', type=float, default=0.2,
                       help='Ratio of data to use for faster search')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--temperature-range', type=float, nargs='+',
                       default=[2.0, 3.0, 4.0, 5.0, 6.0],
                       help='Temperature values to try')
    parser.add_argument('--alpha-range', type=float, nargs='+',
                       default=[0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Alpha values to try')
    
    args = parser.parse_args()
    
    # Run grid search
    results_dict = hyperparameter_grid_search(
        teacher_checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        temperature_range=args.temperature_range,
        alpha_range=args.alpha_range,
        attention_type=args.attention_type,
        epochs=args.epochs,
        subset_ratio=args.subset_ratio,
        device=args.device
    )
    
    # Perform sensitivity analysis
    analyze_sensitivity(
        results=np.array(results_dict['results']),
        temperature_range=results_dict['temperature_range'],
        alpha_range=results_dict['alpha_range'],
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("Hyperparameter interaction analysis completed!")
    print(f"All results saved to {args.output_dir}")
    print("="*80 + "\n")












