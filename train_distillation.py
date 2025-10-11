"""
知识蒸馏训练脚本
支持多种注意力机制的MobileNetV3学生模型
教师模型：ResNet-50
"""

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
import numpy as np

from models import create_model


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    结合硬标签(CE)和软标签(KL散度)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失(与真实标签的交叉熵)
        ce = self.ce_loss(student_logits, labels)
        
        # 软标签损失(与教师输出的KL散度)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 组合损失
        loss = self.alpha * ce + (1 - self.alpha) * kl
        
        return loss, ce.item(), kl.item()


def get_data_loaders(data_dir, batch_size, num_workers=4):
    """
    获取数据加载器
    """
    # 数据预处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 加载数据集
    try:
        # 尝试加载Food-101
        train_dataset = datasets.Food101(
            root=data_dir, split='train',
            transform=train_transform, download=False
        )
        val_dataset = datasets.Food101(
            root=data_dir, split='test',
            transform=val_transform, download=False
        )
        num_classes = 101
        print(f"[INFO] 使用 Food-101 数据集 (101 类)")
    except:
        # 尝试ImageFolder格式
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        print(f"[INFO] 使用 ImageFolder 数据集 ({num_classes} 类)")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, num_classes


def create_teacher_model(num_classes, device):
    """
    创建教师模型(ResNet-50)
    """
    print("[INFO] 创建教师模型: ResNet-50")
    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    teacher = teacher.to(device)
    return teacher


def train_one_epoch(student, teacher, train_loader, criterion, optimizer, 
                    scaler, device, scheduler=None):
    """
    训练一个epoch
    """
    student.train()
    teacher.eval()
    
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 混合精度训练
        with autocast():
            # 教师模型推理(不需要梯度)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # 学生模型推理
            student_logits = student(inputs)
            
            # 计算损失
            loss, ce, kd = criterion(student_logits, teacher_logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_ce += ce * inputs.size(0)
        running_kd += kd * inputs.size(0)
        
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_ce = running_ce / total
    epoch_kd = running_kd / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_ce, epoch_kd, epoch_acc


@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    评估模型
    """
    model.eval()
    correct = 0
    total = 0
    
    for inputs, labels in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(inputs)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("[INFO] 加载数据集...")
    train_loader, val_loader, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"[INFO] 训练集大小: {len(train_loader.dataset)}")
    print(f"[INFO] 验证集大小: {len(val_loader.dataset)}")
    
    # 创建教师模型
    teacher = create_teacher_model(num_classes, device)
    
    # 加载教师模型权重(如果提供)
    if args.teacher_checkpoint:
        print(f"[INFO] 加载教师模型权重: {args.teacher_checkpoint}")
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
        teacher_acc = evaluate(teacher, val_loader, device)
        print(f"[INFO] 教师模型验证准确率: {teacher_acc:.2f}%")
    else:
        print("[WARN] 未提供教师模型权重，将使用随机初始化的教师模型（不推荐）")
    
    # 创建学生模型
    print(f"[INFO] 创建学生模型: MobileNetV3-{args.model_size} + {args.attention_type.upper()}")
    student = create_model(
        num_classes=num_classes,
        attention_type=args.attention_type,
        pretrained=args.pretrained,
        model_size=args.model_size
    ).to(device)
    
    # 打印模型信息
    model_info = student.get_model_info()
    print(f"[INFO] 学生模型参数量: {model_info['total_params']:,}")
    print(f"[INFO] 注意力模块参数: {model_info['attention_params']:,} ({model_info['attention_ratio']})")
    
    # 损失函数
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    
    # 优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            student.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            student.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            student.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_ce': [], 'train_kd': [],
        'train_acc': [], 'val_acc': []
    }
    
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_ce, train_kd, train_acc = train_one_epoch(
            student, teacher, train_loader, criterion, optimizer,
            scaler, device, scheduler if args.scheduler == 'onecycle' else None
        )
        
        # 验证
        val_acc = evaluate(student, val_loader, device)
        
        # 调度器步进(非OneCycle)
        if scheduler is not None and args.scheduler != 'onecycle':
            scheduler.step()
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_ce'].append(train_ce)
        history['train_kd'].append(train_kd)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {train_loss:.4f} (CE: {train_ce:.4f}, KD: {train_kd:.4f}) | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, 
                f'best_mobilenetv3_{args.attention_type}.pth')
            torch.save(student.state_dict(), save_path)
            print(f"  → 保存最佳模型 (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "="*70)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("="*70 + "\n")
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 
        f'history_mobilenetv3_{args.attention_type}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存配置
    config = vars(args)
    config['best_val_acc'] = best_acc
    config['model_info'] = model_info
    config_path = os.path.join(args.output_dir, 
        f'config_mobilenetv3_{args.attention_type}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] 训练历史保存至: {history_path}")
    print(f"[INFO] 配置文件保存至: {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='知识蒸馏训练脚本')
    
    # 数据相关
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 模型相关
    parser.add_argument('--attention-type', type=str, default='eca',
                        choices=['eca', 'simam', 'cbam', 'se', 'coord', 'none'],
                        help='注意力机制类型')
    parser.add_argument('--model-size', type=str, default='large',
                        choices=['large', 'small'],
                        help='MobileNetV3大小')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用ImageNet预训练权重')
    parser.add_argument('--teacher-checkpoint', type=str, default='',
                        help='教师模型权重路径')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器')
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['onecycle', 'cosine', 'none'],
                        help='学习率调度器')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # 蒸馏相关
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='硬标签权重')
    
    # 其他
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)


