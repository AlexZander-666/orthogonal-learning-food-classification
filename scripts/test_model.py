"""
测试训练好的模型
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

from models import create_model


def load_class_names(file_path='class_names.json'):
    """加载类别名称"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        # Food-101类别名称
        return [f"class_{i}" for i in range(101)]


def preprocess_image(image_path):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, class_names, device, top_k=5):
    """预测图像类别"""
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                'class': class_names[idx.item()],
                'probability': prob.item()
            })
    
    return results


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载模型
    print(f"[INFO] 加载模型: {args.model_path}")
    model = create_model(
        num_classes=args.num_classes,
        attention_type=args.attention_type,
        pretrained=False,
        model_size=args.model_size
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # 加载类别名称
    class_names = load_class_names(args.class_names)
    
    # 预处理图像
    print(f"[INFO] 加载图像: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # 预测
    print("\n" + "="*70)
    print("预测结果")
    print("="*70)
    
    results = predict(model, image_tensor, class_names, device, args.top_k)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']:<30} {result['probability']*100:>6.2f}%")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--attention-type', type=str, default='eca',
                        choices=['eca', 'simam', 'cbam', 'se', 'coord', 'none'],
                        help='注意力机制类型')
    parser.add_argument('--model-size', type=str, default='large',
                        choices=['large', 'small'],
                        help='MobileNetV3大小')
    parser.add_argument('--num-classes', type=int, default=101,
                        help='类别数')
    parser.add_argument('--class-names', type=str, default='class_names.json',
                        help='类别名称文件')
    parser.add_argument('--top-k', type=int, default=5,
                        help='返回Top-K预测结果')
    
    args = parser.parse_args()
    
    main(args)


