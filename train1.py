import torch, torchvision, tqdm, os, ssl
from torch.amp import autocast, GradScaler
from models.mobilenetv3_ecb import MobileNetV3_ECA

# ---------- 1. 离线教师 ----------
def get_teacher(device):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(2048, 100)
    return model.to(device)

# ---------- 2. 蒸馏损失 ----------
class DistillLoss(torch.nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T, self.alpha = T, alpha
        self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, s, t, y):
        ce = self.ce(s, y)
        kl = self.kl(torch.log_softmax(s/self.T, dim=1),
                     torch.softmax(t/self.T, dim=1)) * (self.T ** 2)
        return self.alpha * ce + (1 - self.alpha) * kl

# ---------- 3. 训练（梯度累积 + 混合精度） ----------
def train(student, teacher, loader, opt, crit, device, scaler, accum=4):
    student.train(); teacher.eval()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast('cuda'):
            with torch.no_grad():
                t_out = teacher(x)
            s_out = student(x)
            loss = crit(s_out, t_out, y) / accum
        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

# ---------- 4. 评估 ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100 * correct / total

# ---------- 5. DataLoader（Windows 安全版） ----------
def get_loaders(bs):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.507, 0.487, 0.441],
                                         [0.267, 0.256, 0.276])
    ])
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=False, transform=transform)
    # Windows 单进程：num_workers=0，prefetch_factor 必须 None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, persistent_workers=False, prefetch_factor=None)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=True)
    return train_loader, test_loader

# ---------- 6. 主函数 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024          # 5060Ti 16G 能吃满
    train_loader, test_loader = get_loaders(batch_size)

    student = MobileNetV3_ECA(num_classes=100).to(device)
    teacher = get_teacher(device)

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=60)
    criterion = DistillLoss(T=4.0, alpha=0.7)
    scaler = GradScaler('cuda')

    best = 0.0
    for epoch in range(60):
        train(student, teacher, train_loader, optimizer, criterion, device, scaler)
        acc = evaluate(student, test_loader, device)
        print(f"Epoch {epoch:02d} | Acc={acc:.2f}% | LR={scheduler.get_last_lr()[0]:.2e}")
        if acc > best:
            best = acc
            torch.save(student.state_dict(), "best_student.pth")
        scheduler.step()
    print(f"Best accuracy: {best:.2f}%")

if __name__ == "__main__":
    main()