"""
训练教师模型(ResNet-50)
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models


def get_data_loaders(data_dir, batch_size, num_workers=4):
    """获取数据加载器"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        train_dataset = datasets.Food101(
            root=data_dir, split='train',
            transform=train_transform, download=False
        )
        val_dataset = datasets.Food101(
            root=data_dir, split='test',
            transform=val_transform, download=False
        )
        num_classes = 101
        print(f"[INFO] 使用 Food-101 数据集")
    except:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        print(f"[INFO] 使用 ImageFolder 数据集")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, num_classes


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载数据
    print("[INFO] 加载数据集...")
    train_loader, val_loader, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"[INFO] 训练集大小: {len(train_loader.dataset)}")
    print(f"[INFO] 验证集大小: {len(val_loader.dataset)}")
    print(f"[INFO] 类别数: {num_classes}")
    
    # 创建模型
    print("[INFO] 创建 ResNet-50 教师模型...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练
    best_acc = 0.0
    
    print("\n" + "="*70)
    print("开始训练教师模型")
    print("="*70)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 调度器步进
        scheduler.step()
        
        # 打印
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  → 保存最佳模型 (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "="*70)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存至: {args.output}")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练ResNet-50教师模型')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output', type=str, default='teacher_resnet50.pth',
                        help='输出模型路径')
    
    args = parser.parse_args()
    
    main(args)


