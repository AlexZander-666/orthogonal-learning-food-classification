#!/usr/bin/env python3
"""
统计显著性检验脚本
进行多次独立实验，计算置信区间和t检验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
from scipy import stats
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3_attention import MobileNetV3WithAttention


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir='./data', batch_size=64, num_workers=4):
    """Create data loaders for Food-101"""
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
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
    
    test_dataset = datasets.Food101(
        root=data_dir,
        split='test',
        transform=transform_test,
        download=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_one_run(
    model,
    train_loader,
    test_loader,
    epochs=30,
    lr=0.05,
    device='cuda',
    verbose=False
):
    """Train model for one run and return test accuracy"""
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if verbose:
            print(f'Epoch {epoch+1}: Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)')
    
    return best_acc


def train_with_kd_one_run(
    student,
    teacher,
    train_loader,
    test_loader,
    temperature=4.0,
    alpha=0.7,
    epochs=30,
    lr=0.05,
    device='cuda',
    verbose=False
):
    """Train student with KD for one run and return test accuracy"""
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    
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
        
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            # Student forward
            student_logits = student(images)
            
            # Teacher forward
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
        
        # Test
        student.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if verbose:
            print(f'Epoch {epoch+1}: Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)')
    
    return best_acc


def run_multiple_experiments(
    model_type,
    num_runs=10,
    data_dir='./data',
    teacher_checkpoint=None,
    attention_type=None,
    use_kd=False,
    temperature=4.0,
    alpha=0.7,
    epochs=30,
    device='cuda',
    output_dir='./experiments/statistical_results'
):
    """
    Run multiple independent experiments with different random seeds
    
    Args:
        model_type: 'baseline', 'teacher', or 'student'
        num_runs: Number of independent runs
        data_dir: Path to dataset
        teacher_checkpoint: Path to teacher checkpoint (for KD)
        attention_type: Attention mechanism type
        use_kd: Whether to use knowledge distillation
        temperature: KD temperature
        alpha: KD loss weight
        epochs: Training epochs
        device: Device to use
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Running {num_runs} independent experiments for {model_type}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_loader, test_loader = get_data_loaders(data_dir)
    
    # Load teacher if needed
    teacher = None
    if use_kd and teacher_checkpoint:
        print("Loading teacher model...")
        teacher = models.resnet50()
        teacher.classifier = nn.Linear(teacher.classifier.in_features, 101)
        checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher = teacher.to(device)
        teacher.eval()
    
    # Run experiments
    accuracies = []
    seeds = [42 + i * 100 for i in range(num_runs)]  # Different seeds for each run
    
    for run_idx, seed in enumerate(seeds):
        print(f"\nRun {run_idx + 1}/{num_runs} (seed={seed})")
        print("-" * 40)
        
        # Set seed
        set_seed(seed)
        
        # Create model
        if model_type == 'baseline':
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 101)
        elif model_type == 'teacher':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 101)
        else:  # student
            model = MobileNetV3WithAttention(
                num_classes=101,
                attention_type=attention_type,
                pretrained=True
            )
        
        # Train
        if use_kd:
            acc = train_with_kd_one_run(
                student=model,
                teacher=teacher,
                train_loader=train_loader,
                test_loader=test_loader,
                temperature=temperature,
                alpha=alpha,
                epochs=epochs,
                device=device,
                verbose=(run_idx == 0)  # Only show progress for first run
            )
        else:
            acc = train_one_run(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=epochs,
                device=device,
                verbose=(run_idx == 0)
            )
        
        accuracies.append(acc)
        print(f"Test Accuracy: {acc:.2f}%")
    
    # Compute statistics
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample std
    sem_acc = stats.sem(accuracies)  # Standard error of mean
    
    # 95% confidence interval
    ci_95 = stats.t.interval(
        0.95,
        len(accuracies) - 1,
        loc=mean_acc,
        scale=sem_acc
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Statistical Summary for {model_type}")
    print(f"{'='*80}")
    print(f"Number of runs:          {num_runs}")
    print(f"Mean accuracy:           {mean_acc:.2f}%")
    print(f"Standard deviation:      {std_acc:.2f}%")
    print(f"Standard error:          {sem_acc:.2f}%")
    print(f"95% CI:                  [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]")
    print(f"Min accuracy:            {accuracies.min():.2f}%")
    print(f"Max accuracy:            {accuracies.max():.2f}%")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'model_type': model_type,
        'num_runs': num_runs,
        'seeds': seeds,
        'accuracies': accuracies.tolist(),
        'mean': float(mean_acc),
        'std': float(std_acc),
        'sem': float(sem_acc),
        'ci_95': [float(ci_95[0]), float(ci_95[1])],
        'min': float(accuracies.min()),
        'max': float(accuracies.max())
    }
    
    output_file = os.path.join(output_dir, f'{model_type}_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return results


def compare_models_ttest(results1, results2, model1_name, model2_name, output_dir):
    """
    Perform t-test to compare two models
    
    Args:
        results1: Statistics results for model 1
        results2: Statistics results for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        output_dir: Directory to save results
    """
    acc1 = np.array(results1['accuracies'])
    acc2 = np.array(results2['accuracies'])
    
    # Paired t-test (if same number of runs)
    if len(acc1) == len(acc2):
        t_stat, p_value = stats.ttest_rel(acc2, acc1)
        test_type = "Paired t-test"
    else:
        t_stat, p_value = stats.ttest_ind(acc2, acc1)
        test_type = "Independent t-test"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(acc1, ddof=1) + np.var(acc2, ddof=1)) / 2)
    cohens_d = (results2['mean'] - results1['mean']) / pooled_std
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Statistical Comparison: {model2_name} vs {model1_name}")
    print(f"{'='*80}")
    print(f"{model1_name}: {results1['mean']:.2f}% ± {results1['std']:.2f}%")
    print(f"{model2_name}: {results2['mean']:.2f}% ± {results2['std']:.2f}%")
    print(f"\nMean difference:         {results2['mean'] - results1['mean']:.2f}%")
    print(f"Relative improvement:    {(results2['mean'] - results1['mean']) / results1['mean'] * 100:.1f}%")
    print(f"\n{test_type}:")
    print(f"  t-statistic:           {t_stat:.4f}")
    print(f"  p-value:               {p_value:.6f}")
    print(f"  Significant (α=0.05):  {'Yes' if p_value < 0.05 else 'No'}")
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    print(f"  Interpretation:        {interpret_cohens_d(cohens_d)}")
    print(f"{'='*80}\n")
    
    # Save comparison
    comparison = {
        'model1': model1_name,
        'model2': model2_name,
        'model1_mean': results1['mean'],
        'model2_mean': results2['mean'],
        'mean_difference': results2['mean'] - results1['mean'],
        'relative_improvement': (results2['mean'] - results1['mean']) / results1['mean'] * 100,
        'test_type': test_type,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'effect_size_interpretation': interpret_cohens_d(cohens_d)
    }
    
    output_file = os.path.join(output_dir, f'comparison_{model2_name}_vs_{model1_name}.json')
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {output_file}")
    
    return comparison


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def visualize_results(results_dict, output_dir):
    """Create visualization of multiple model results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    data = []
    labels = []
    for name, results in results_dict.items():
        data.append(results['accuracies'])
        labels.append(name)
    
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette('Set2', len(data))):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Distribution of Test Accuracies Across Multiple Runs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bar plot with error bars
    x = np.arange(len(labels))
    means = [results['mean'] for results in results_dict.values()]
    stds = [results['std'] for results in results_dict.values()]
    
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette('Set2', len(data)))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Mean Test Accuracy (%)', fontsize=12)
    ax2.set_title('Mean Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'statistical_comparison.png')}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run statistical significance experiments'
    )
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to Food-101 dataset')
    parser.add_argument('--output-dir', type=str, default='./experiments/statistical_results',
                       help='Directory to save results')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of independent runs')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs per run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                       help='Path to teacher model checkpoint')
    
    args = parser.parse_args()
    
    # Run experiments for different configurations
    results = {}
    
    print("Running statistical experiments...")
    print(f"Number of runs per configuration: {args.num_runs}")
    print(f"Epochs per run: {args.epochs}\n")
    
    # 1. Baseline
    print("\n[1/4] Baseline MobileNetV3...")
    results['baseline'] = run_multiple_experiments(
        model_type='baseline',
        num_runs=args.num_runs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 2. Teacher
    print("\n[2/4] ResNet-50 Teacher...")
    results['teacher'] = run_multiple_experiments(
        model_type='teacher',
        num_runs=args.num_runs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 3. SimAM without KD
    print("\n[3/4] MobileNetV3 + SimAM...")
    results['simam'] = run_multiple_experiments(
        model_type='student',
        attention_type='simam',
        use_kd=False,
        num_runs=args.num_runs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 4. SimAM with KD
    if args.teacher_checkpoint:
        print("\n[4/4] MobileNetV3 + SimAM + KD...")
        results['simam_kd'] = run_multiple_experiments(
            model_type='student',
            attention_type='simam',
            use_kd=True,
            teacher_checkpoint=args.teacher_checkpoint,
            num_runs=args.num_runs,
            data_dir=args.data_dir,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir
        )
    
    # Perform comparisons
    print("\n" + "="*80)
    print("Statistical Comparisons")
    print("="*80)
    
    # SimAM vs Baseline
    compare_models_ttest(
        results['baseline'],
        results['simam'],
        'Baseline',
        'SimAM',
        args.output_dir
    )
    
    # SimAM+KD vs Baseline
    if 'simam_kd' in results:
        compare_models_ttest(
            results['baseline'],
            results['simam_kd'],
            'Baseline',
            'SimAM+KD',
            args.output_dir
        )
        
        # SimAM+KD vs Teacher
        compare_models_ttest(
            results['teacher'],
            results['simam_kd'],
            'Teacher',
            'SimAM+KD',
            args.output_dir
        )
    
    # Create visualizations
    visualize_results(results, args.output_dir)
    
    print("\n" + "="*80)
    print("Statistical experiments completed!")
    print(f"All results saved to {args.output_dir}")
    print("="*80 + "\n")












