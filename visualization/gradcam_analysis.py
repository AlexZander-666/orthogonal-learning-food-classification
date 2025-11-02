#!/usr/bin/env python3
"""
Grad-CAM可视化分析脚本
用于对比基线、SimAM和SimAM+KD模型的激活模式
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3_attention import MobileNetV3WithAttention


class GradCAM:
    """Grad-CAM implementation for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activation"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradient"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Calculate weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def load_model(checkpoint_path, attention_type=None, num_classes=101):
    """Load trained model from checkpoint"""
    if attention_type:
        model = MobileNetV3WithAttention(
            num_classes=num_classes,
            attention_type=attention_type
        )
    else:
        # Baseline without attention
        import torchvision.models as models
        model = models.mobilenet_v3_large(num_classes=num_classes)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


def visualize_gradcam_comparison(
    image_path,
    models_dict,
    save_path,
    class_names=None,
    target_class=None
):
    """
    Visualize Grad-CAM for multiple models on the same image
    
    Args:
        image_path: Path to input image
        models_dict: Dict of {model_name: (model, target_layer)}
        save_path: Path to save visualization
        class_names: List of class names
        target_class: Target class index (None for predicted class)
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0)
    
    # Prepare figure
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))
    
    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Generate and visualize Grad-CAM for each model
    for idx, (model_name, (model, target_layer)) in enumerate(models_dict.items(), 1):
        # Create Grad-CAM
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor, target_class)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Upsample CAM to match image size
        cam_resized = np.array(Image.fromarray(cam).resize(
            original_image.size, Image.BILINEAR
        ))
        
        # Show heatmap
        axes[0, idx].imshow(original_image)
        axes[0, idx].imshow(cam_resized, cmap='jet', alpha=0.5)
        
        title = f'{model_name}\n'
        if class_names and pred_class < len(class_names):
            title += f'{class_names[pred_class]}\n'
        title += f'Conf: {confidence:.2%}'
        axes[0, idx].set_title(title, fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Show pure heatmap
        axes[1, idx].imshow(cam_resized, cmap='jet')
        axes[1, idx].set_title(f'{model_name} (Heatmap Only)', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")


def run_gradcam_experiments(
    data_dir='./data',
    output_dir='./visualization/gradcam_results',
    baseline_checkpoint=None,
    simam_checkpoint=None,
    simam_kd_checkpoint=None,
    num_samples=10
):
    """
    Run Grad-CAM visualization experiments
    
    Args:
        data_dir: Path to Food-101 dataset
        output_dir: Directory to save visualizations
        baseline_checkpoint: Path to baseline model checkpoint
        simam_checkpoint: Path to SimAM model checkpoint
        simam_kd_checkpoint: Path to SimAM+KD model checkpoint
        num_samples: Number of sample images to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    baseline_model = load_model(baseline_checkpoint, attention_type=None)
    simam_model = load_model(simam_checkpoint, attention_type='simam')
    simam_kd_model = load_model(simam_kd_checkpoint, attention_type='simam')
    
    # Set to evaluation mode
    baseline_model.eval()
    simam_model.eval()
    simam_kd_model.eval()
    
    # Get target layers (last convolutional layer before classifier)
    def get_target_layer(model):
        if hasattr(model, 'features'):
            # MobileNetV3
            return model.features[-1]
        else:
            # Custom model with attention
            return model.backbone.features[-1]
    
    baseline_layer = get_target_layer(baseline_model)
    simam_layer = get_target_layer(simam_model)
    simam_kd_layer = get_target_layer(simam_kd_model)
    
    models_dict = {
        'Baseline': (baseline_model, baseline_layer),
        'SimAM': (simam_model, simam_layer),
        'SimAM+KD': (simam_kd_model, simam_kd_layer)
    }
    
    # Load dataset to get class names
    test_dataset = datasets.Food101(
        root=data_dir,
        split='test',
        download=False
    )
    class_names = test_dataset.classes
    
    # Select sample images (diversely from different classes)
    sample_indices = np.linspace(0, len(test_dataset) - 1, num_samples, dtype=int)
    
    print(f"Generating Grad-CAM visualizations for {num_samples} samples...")
    for i, idx in enumerate(sample_indices):
        image_path, label = test_dataset.samples[idx]
        class_name = class_names[label]
        
        save_path = os.path.join(
            output_dir,
            f'gradcam_{i:02d}_{class_name.replace("/", "_")}.png'
        )
        
        print(f"Processing {i+1}/{num_samples}: {class_name}...")
        visualize_gradcam_comparison(
            image_path=image_path,
            models_dict=models_dict,
            save_path=save_path,
            class_names=class_names,
            target_class=label
        )
    
    print(f"\nAll visualizations saved to {output_dir}")
    
    # Generate summary statistics
    generate_summary_report(output_dir, num_samples)


def generate_summary_report(output_dir, num_samples):
    """Generate a summary report of the Grad-CAM analysis"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Grad-CAM Visualization Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total samples visualized: {num_samples}\n\n")
        
        f.write("Models compared:\n")
        f.write("  1. Baseline (MobileNetV3 without attention)\n")
        f.write("  2. SimAM (MobileNetV3 + SimAM attention)\n")
        f.write("  3. SimAM+KD (MobileNetV3 + SimAM + Knowledge Distillation)\n\n")
        
        f.write("Key observations to analyze:\n")
        f.write("  - Baseline vs SimAM: Does attention focus on more discriminative regions?\n")
        f.write("  - SimAM vs SimAM+KD: Does KD lead to more refined activations?\n")
        f.write("  - Localization quality: Do models focus on food items vs background?\n")
        f.write("  - Activation confidence: Are heatmaps more concentrated or diffuse?\n\n")
        
        f.write("Expected hypothesis:\n")
        f.write("  1. SimAM should show more focused attention on food components\n")
        f.write("  2. SimAM+KD should show even more refined and confident activations\n")
        f.write("  3. Baseline may have more scattered attention patterns\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Summary report saved to {report_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for model comparison'
    )
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to Food-101 dataset')
    parser.add_argument('--output-dir', type=str, default='./visualization/gradcam_results',
                       help='Directory to save visualizations')
    parser.add_argument('--baseline-checkpoint', type=str, default=None,
                       help='Path to baseline model checkpoint')
    parser.add_argument('--simam-checkpoint', type=str, default=None,
                       help='Path to SimAM model checkpoint')
    parser.add_argument('--simam-kd-checkpoint', type=str, default=None,
                       help='Path to SimAM+KD model checkpoint')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of sample images to visualize')
    
    args = parser.parse_args()
    
    run_gradcam_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        simam_checkpoint=args.simam_checkpoint,
        simam_kd_checkpoint=args.simam_kd_checkpoint,
        num_samples=args.num_samples
    )












