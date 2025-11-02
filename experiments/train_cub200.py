#!/usr/bin/env python3
"""
CUB-200-2011数据集上的泛化验证实验
验证方法在食品以外的细粒度分类任务上的有效性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3_attention import MobileNetV3WithAttention


class CUB200Dataset(Dataset):
    """CUB-200-2011 Dataset"""
    
    def __init__(self, root_dir, split='train', transform=None, download=False):
        """
        Args:
            root_dir: Root directory of CUB-200-2011 dataset
            split: 'train' or 'test'
            transform: Image transformations
            download: Whether to download the dataset
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Download if needed
        if download and not self.root_dir.exists():
            self.download_dataset()
        
        # Load metadata
        self.images_path = self.root_dir / 'images.txt'
        self.labels_path = self.root_dir / 'image_class_labels.txt'
        self.split_path = self.root_dir / 'train_test_split.txt'
        self.classes_path = self.root_dir / 'classes.txt'
        
        # Parse files
        self.images = {}  # id -> path
        self.labels = {}  # id -> class_id
        self.splits = {}  # id -> is_train (1=train, 0=test)
        self.classes = {}  # class_id -> class_name
        
        self._load_metadata()
        
        # Filter by split
        self.data = []
        split_flag = 1 if split == 'train' else 0
        for img_id, is_train in self.splits.items():
            if is_train == split_flag:
                img_path = self.root_dir / 'images' / self.images[img_id]
                label = self.labels[img_id] - 1  # Convert to 0-indexed
                self.data.append((str(img_path), label))
    
    def _load_metadata(self):
        """Load metadata files"""
        # Load images
        with open(self.images_path) as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.images[img_id] = img_path
        
        # Load labels
        with open(self.labels_path) as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.labels[img_id] = int(class_id)
        
        # Load split
        with open(self.split_path) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                self.splits[img_id] = int(is_train)
        
        # Load class names
        with open(self.classes_path) as f:
            for line in f:
                class_id, class_name = line.strip().split(' ', 1)
                self.classes[int(class_id)] = class_name
    
    def download_dataset(self):
        """Download CUB-200-2011 dataset"""
        print("CUB-200-2011 dataset not found.")
        print("Please download manually from:")
        print("https://data.caltech.edu/records/65de6-vp158")
        print(f"Extract to: {self.root_dir}")
        raise FileNotFoundError("Dataset not found. Please download manually.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """Create data loaders for CUB-200-2011"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CUB200Dataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    test_dataset = CUB200Dataset(
        root_dir=data_dir,
        split='test',
        transform=val_transform
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


def train_baseline(
    data_dir,
    output_dir,
    epochs=30,
    batch_size=64,
    lr=0.05,
    device='cuda'
):
    """Train baseline MobileNetV3 on CUB-200"""
    print("\n" + "="*80)
    print("Training Baseline MobileNetV3 on CUB-200-2011")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 200)
    model = model.to(device)
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
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
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        train_acc = 100. * train_correct / train_total
        
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
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        results['train_loss'].append(train_loss / len(train_loader))
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(output_dir, 'baseline_best.pth'))
    
    # Save results
    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    return best_acc


def train_teacher(
    data_dir,
    output_dir,
    epochs=30,
    batch_size=64,
    lr=0.01,
    device='cuda'
):
    """Train ResNet-50 teacher on CUB-200"""
    print("\n" + "="*80)
    print("Training ResNet-50 Teacher on CUB-200-2011")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = model.to(device)
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
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
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        train_acc = 100. * train_correct / train_total
        
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
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        results['train_loss'].append(train_loss / len(train_loader))
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(output_dir, 'teacher_best.pth'))
    
    # Save results
    with open(os.path.join(output_dir, 'teacher_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    return best_acc


def train_student_with_kd(
    data_dir,
    teacher_checkpoint,
    output_dir,
    attention_type='simam',
    temperature=4.0,
    alpha=0.7,
    epochs=30,
    batch_size=64,
    lr=0.05,
    device='cuda'
):
    """Train student with knowledge distillation on CUB-200"""
    print("\n" + "="*80)
    print(f"Training MobileNetV3+{attention_type.upper()}+KD on CUB-200-2011")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load teacher
    teacher = models.resnet50()
    teacher.fc = nn.Linear(teacher.fc.in_features, 200)
    checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher = teacher.to(device)
    teacher.eval()
    
    # Create student
    student = MobileNetV3WithAttention(
        num_classes=200,
        attention_type=attention_type,
        pretrained=True
    )
    student = student.to(device)
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
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
    
    # Training loop
    best_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        student.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
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
            
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        train_acc = 100. * train_correct / train_total
        
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
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        results['train_loss'].append(train_loss / len(train_loader))
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(output_dir, f'student_{attention_type}_kd_best.pth'))
    
    # Save results
    with open(os.path.join(output_dir, f'student_{attention_type}_kd_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    return best_acc


def run_cub_generalization_study(
    data_dir='./data/CUB_200_2011',
    output_dir='./experiments/cub200_results',
    device='cuda'
):
    """Run complete CUB-200 generalization study"""
    print("\n" + "="*80)
    print("CUB-200-2011 Generalization Study")
    print("="*80 + "\n")
    
    results = {}
    
    # 1. Train baseline
    print("\n[1/3] Training baseline MobileNetV3...")
    baseline_acc = train_baseline(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device
    )
    results['baseline'] = baseline_acc
    
    # 2. Train teacher
    print("\n[2/3] Training ResNet-50 teacher...")
    teacher_acc = train_teacher(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device
    )
    results['teacher'] = teacher_acc
    
    # 3. Train student with KD
    print("\n[3/3] Training MobileNetV3+SimAM+KD...")
    teacher_checkpoint = os.path.join(output_dir, 'teacher_best.pth')
    student_acc = train_student_with_kd(
        data_dir=data_dir,
        teacher_checkpoint=teacher_checkpoint,
        output_dir=output_dir,
        attention_type='simam',
        device=device
    )
    results['student_simam_kd'] = student_acc
    
    # Print summary
    print("\n" + "="*80)
    print("CUB-200-2011 Generalization Results Summary")
    print("="*80)
    print(f"Baseline MobileNetV3:        {results['baseline']:.2f}%")
    print(f"ResNet-50 Teacher:           {results['teacher']:.2f}%")
    print(f"MobileNetV3+SimAM+KD:        {results['student_simam_kd']:.2f}%")
    print(f"\nImprovement over baseline:   {results['student_simam_kd'] - results['baseline']:.2f}%")
    print(f"Relative improvement:        {(results['student_simam_kd'] - results['baseline']) / results['baseline'] * 100:.1f}%")
    print("="*80 + "\n")
    
    # Save summary
    with open(os.path.join(output_dir, 'cub200_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CUB-200-2011 Generalization Validation'
    )
    parser.add_argument('--data-dir', type=str, default='./data/CUB_200_2011',
                       help='Path to CUB-200-2011 dataset')
    parser.add_argument('--output-dir', type=str, default='./experiments/cub200_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run generalization study
    run_cub_generalization_study(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )












