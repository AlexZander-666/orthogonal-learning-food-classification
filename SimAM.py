# train.py (优化版：加入 teacher logits 预计算以提高 GPU 利用率)
import argparse, os, time, platform, multiprocessing, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets.folder import default_loader

# ---------------- SimAM (用于 student) ----------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        xm = x.mean(dim=[2,3], keepdim=True)
        x_minus_mu_square = (x - xm).pow(2)
        denom = 4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)
        y = x_minus_mu_square / denom + 0.5
        return x * y.sigmoid()

# ---------------- Student model ----------------
class MobileNetV3_ECA(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=None)
        self.features = mobilenet.features
        last_ch = mobilenet.classifier[0].in_features
        self.simam = SimAM()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_ch, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.simam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------- small dataset helpers ----------------
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform=None, loader=default_loader):
        self.items = items
        self.transform = transform
        self.loader = loader
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def _read_food101_meta_items(images_dir, meta_txt_path, class_to_idx):
    items = []
    if not os.path.isfile(meta_txt_path):
        raise FileNotFoundError(f"meta file not found: {meta_txt_path}")
    with open(meta_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip()
            if not rel: continue
            rel_os = rel.replace('/', os.sep).replace('\\', os.sep)
            cand_paths = []
            if os.path.splitext(rel_os)[1] != '':
                cand_paths.append(os.path.join(images_dir, rel_os))
            else:
                for ext in ('.jpg','.JPG','.jpeg','.png'):
                    cand_paths.append(os.path.join(images_dir, rel_os + ext))
            found=False
            for p in cand_paths:
                if os.path.isfile(p):
                    cls = rel_os.split(os.sep)[0]
                    if cls not in class_to_idx:
                        found=True; break
                    items.append((p, class_to_idx[cls])); found=True; break
            if not found:
                # fallback tries
                parts = rel_os.split(os.sep)
                cls = parts[0]; name = parts[-1]
                candidate = os.path.join(images_dir, cls, name)
                if os.path.isfile(candidate): items.append((candidate, class_to_idx.get(cls,-1)))
                elif os.path.isfile(candidate + '.jpg'): items.append((candidate + '.jpg', class_to_idx.get(cls,-1)))
                else:
                    print(f"[WARN] cannot find image for meta entry: '{rel}'")
    return items

def find_food101_root(base_dir):
    cand1 = os.path.join(base_dir, 'images')
    cand2 = os.path.join(base_dir, 'food-101', 'images')
    if os.path.isdir(cand1): return base_dir
    if os.path.isdir(cand2): return os.path.join(base_dir, 'food-101')
    return None

# ---------------- robust loaders + auto workers ----------------
def get_loaders_safe(base_dir, batch_size, num_workers=None):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(), normalize])

    cpu_count = multiprocessing.cpu_count()
    sysname = platform.system()
    if num_workers is None:
        if sysname == 'Windows' or sysname == 'Darwin':
            num_workers = max(2, min(6, cpu_count - 1))
            persistent_workers=False; prefetch_factor=None
        else:
            num_workers = max(4, min(12, cpu_count - 2))
            persistent_workers=True; prefetch_factor=2
    else:
        persistent_workers = (num_workers > 0 and platform.system() != 'Windows')
        prefetch_factor = 2 if num_workers > 0 else None

    food_root = find_food101_root(base_dir)
    if food_root is not None:
        try:
            print(f"[INFO] Attempting torchvision Food-101 at: {food_root}")
            train_dataset = datasets.Food101(root=food_root, split='train', transform=train_transform, download=False)
            val_dataset = datasets.Food101(root=food_root, split='test', transform=val_transform, download=False)
            print("[INFO] torchvision loaded.")
        except Exception as e:
            print("[WARN] torchvision Food101 failed, fallback. Err:", e)
            images_dir = os.path.join(food_root, 'images')
            meta_dir = os.path.join(food_root, 'meta')
            if not os.path.isdir(images_dir):
                raise RuntimeError(f"images dir not found: {images_dir}")
            classes = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir,d))])
            class_to_idx = {c:i for i,c in enumerate(classes)}
            train_meta = os.path.join(meta_dir, 'train.txt'); test_meta = os.path.join(meta_dir, 'test.txt')
            if not os.path.isfile(train_meta) or not os.path.isfile(test_meta):
                possible = [p for p in (os.path.join(food_root,'meta'), os.path.join(base_dir,'meta')) if os.path.isdir(p)]
                if possible:
                    meta_dir = possible[0]; train_meta = os.path.join(meta_dir,'train.txt'); test_meta = os.path.join(meta_dir,'test.txt')
            if not os.path.isfile(train_meta) or not os.path.isfile(test_meta):
                raise RuntimeError(f"Cannot find meta train/test under {meta_dir}.")
            train_items = _read_food101_meta_items(images_dir, train_meta, class_to_idx)
            test_items = _read_food101_meta_items(images_dir, test_meta, class_to_idx)
            train_dataset = ListDataset(train_items, transform=train_transform)
            val_dataset = ListDataset(test_items, transform=val_transform)
    else:
        tr = os.path.join(base_dir,'train'); va = os.path.join(base_dir,'val')
        if os.path.isdir(tr) and os.path.isdir(va):
            train_dataset = datasets.ImageFolder(tr, transform=train_transform)
            val_dataset = datasets.ImageFolder(va, transform=val_transform)
        else:
            raise RuntimeError("Dataset not found. Place Food-101 images/ or ImageFolder structure.")

    loader_kwargs = dict(batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    if num_workers>0:
        loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None: loader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    val_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    if num_workers>0:
        val_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None: val_kwargs['prefetch_factor'] = prefetch_factor
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    print(f"[INFO] DataLoaders created (num_workers={num_workers}, persistent_workers={persistent_workers})")
    return train_loader, val_loader

# ---------------- DataPrefetcher generalized ----------------
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.done=False
        self.preload()
    def preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            self.done=True
            return
        # support (input,label) or (input,label,logits)
        with torch.cuda.stream(self.stream):
            to_device = lambda x: x.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last) if isinstance(x, torch.Tensor) and x.dim()==4 else (x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
            if isinstance(batch, (list,tuple)):
                self.next_batch = tuple(to_device(x) for x in batch)
            else:
                self.next_batch = to_device(batch)
    def next(self):
        if self.done: return None
        torch.cuda.current_stream().wait_stream(self.stream)
        b = self.next_batch
        self.preload()
        return b

# ---------------- Precompute teacher logits ----------------
def precompute_teacher_logits(teacher, loader, device, out_path):
    # compute logits for loader.dataset in order (loader must not shuffle)
    print(f"[INFO] Precomputing teacher logits -> {out_path} (this may take a while)")
    teacher = teacher.to(device).to(memory_format=torch.channels_last)
    teacher.eval()
    all_logits = []
    all_labels = []
    scaler = GradScaler()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).contiguous(memory_format=torch.channels_last)
            out = teacher(inputs)
            all_logits.append(out.cpu().half())  # save as float16 to save space
            all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'logits': logits, 'labels': labels}, out_path)
    print(f"[INFO] Saved logits: {out_path} (shape={logits.shape})")
    return out_path

# ---------------- Dataset wrapper to attach precomputed logits ----------------
class DatasetWithLogits(torch.utils.data.Dataset):
    def __init__(self, base_dataset, logits_tensor):
        self.base = base_dataset
        self.logits = logits_tensor
        assert len(self.base) == len(self.logits), "dataset and logits length mismatch"
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        tlog = self.logits[idx]
        return img, label, tlog

# ---------------- Training & eval loops (use precomputed logits if provided) ----------------
def train_one_epoch(student, train_loader, optimizer, scaler, device, T, alpha, scheduler=None, use_precomputed=False):
    student.train()
    running_loss = 0.0; total=0
    prefetcher = DataPrefetcher(train_loader, device)
    while True:
        batch = prefetcher.next()
        if batch is None: break
        if use_precomputed:
            inputs, labels, t_logits = batch
            t_logits = t_logits.to(device) if not (isinstance(t_logits, torch.Tensor) and t_logits.device==device) else t_logits
        else:
            inputs, labels = batch
        optimizer.zero_grad()
        with autocast():
            if use_precomputed:
                s_logits = student(inputs)
                ce_loss = F.cross_entropy(s_logits, labels)
                s_log_prob = F.log_softmax(s_logits / T, dim=1)
                t_prob = F.softmax(t_logits.to(s_logits.dtype) / T, dim=1)
                kd_loss = F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (T*T)
                loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
            else:
                # shouldn't be used if precompute option selected; kept for fallback
                raise RuntimeError("train_one_epoch called without precomputed logits. Use train_one_epoch_nonpre() or enable precompute.")
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        if scheduler is not None: scheduler.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    return running_loss / total if total>0 else 0.0

@torch.no_grad()
def evaluate_student(student, val_loader, device, use_precomputed=False, precomp_logits=None):
    student.eval(); correct=0; total=0
    prefetcher = DataPrefetcher(val_loader, device)
    while True:
        batch = prefetcher.next()
        if batch is None: break
        if use_precomputed:
            inputs, labels, t_logits = batch
        else:
            inputs, labels = batch
        outputs = student(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total>0 else 0.0

# ---------------- Quick teacher finetune (optional) ----------------
def finetune_teacher(teacher, train_loader, val_loader, device, epochs=3, lr=1e-3):
    print("[INFO] Quick finetune teacher for", epochs, "epochs")
    teacher = teacher.to(device).to(memory_format=torch.channels_last)
    teacher.train()
    optimizer = optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    for ep in range(epochs):
        t0=time.time(); running=0.0; total=0
        prefetcher = DataPrefetcher(train_loader, device)
        while True:
            batch = prefetcher.next()
            if batch is None: break
            inputs, labels = batch
            optimizer.zero_grad()
            with autocast():
                outputs = teacher(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item() * inputs.size(0); total += inputs.size(0)
        scheduler.step()
        val_acc = evaluate_student(teacher, val_loader, device, use_precomputed=False)
        print(f"[Teacher Finetune] Epoch {ep+1}/{epochs} loss={running/total:.4f} val_acc={val_acc:.2f}% time={time.time()-t0:.1f}s")
    return teacher

# ---------------- Main ----------------
def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    train_loader, val_loader = get_loaders_safe(args.data_dir, args.batch_size, num_workers=args.num_workers)
    if hasattr(train_loader.dataset, 'classes'):
        num_classes = len(train_loader.dataset.classes)
    else:
        num_classes = args.num_classes
    print("[INFO] num_classes =", num_classes)

    # teacher model
    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)

    # load teacher or quick finetune
    if args.teacher_checkpoint and os.path.isfile(args.teacher_checkpoint):
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location='cpu'))
        print("[INFO] Loaded teacher checkpoint.")
    else:
        if args.pretrain_teacher:
            teacher = finetune_teacher(teacher, train_loader, val_loader, device, epochs=args.teacher_epochs, lr=args.teacher_lr)
            torch.save(teacher.state_dict(), args.teacher_ckpt_path)
            print(f"[INFO] Saved quick teacher -> {args.teacher_ckpt_path}")

    # optionally precompute teacher logits
    use_precomputed = False
    if args.precompute_teacher_logits:
        # for precompute we need deterministic loader order -> recreate train/val loaders with shuffle=False
        print("[INFO] Precomputing teacher logits for train & val (shuffle=False).")
        # recreate loaders with shuffle=False and same transforms
        # assume original dataset objects accessible
        tr_dataset = train_loader.dataset
        va_dataset = val_loader.dataset
        pre_batch = max(1, args.precompute_batch_size)
        tr_loader_for_pre = torch.utils.data.DataLoader(tr_dataset, batch_size=pre_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        va_loader_for_pre = torch.utils.data.DataLoader(va_dataset, batch_size=pre_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        pre_train_path = os.path.join(args.data_dir, 'teacher_train_logits.pt')
        pre_val_path = os.path.join(args.data_dir, 'teacher_val_logits.pt')
        precompute_teacher_logits(teacher, tr_loader_for_pre, device, pre_train_path)
        precompute_teacher_logits(teacher, va_loader_for_pre, device, pre_val_path)
        # load back logits to memory (CPU)
        tr_dict = torch.load(pre_train_path, map_location='cpu')
        va_dict = torch.load(pre_val_path, map_location='cpu')
        tr_logits = tr_dict['logits']; va_logits = va_dict['logits']
        # wrap original datasets
        train_dataset_with_logits = DatasetWithLogits(train_loader.dataset, tr_logits)
        val_dataset_with_logits = DatasetWithLogits(val_loader.dataset, va_logits)
        # recreate final loaders (shuffle True for training)
        train_loader = torch.utils.data.DataLoader(train_dataset_with_logits, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_with_logits, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
        use_precomputed = True
        print("[INFO] Using precomputed teacher logits for distillation.")

    # Prepare teacher/student on device
    for p in teacher.parameters(): p.requires_grad = False
    teacher = teacher.to(device).to(memory_format=torch.channels_last)
    teacher.eval()

    student = MobileNetV3_ECA(num_classes=num_classes).to(device).to(memory_format=torch.channels_last)

    optimizer = optim.SGD(student.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=math.ceil(len(train_loader)))
    scaler = GradScaler()
    best_acc = 0.0

    # training loop uses precomputed logits mode
    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(student, train_loader, optimizer, scaler, device, args.temperature, args.alpha, scheduler=scheduler, use_precomputed=use_precomputed)
        val_acc = evaluate_student(student, val_loader, device, use_precomputed=use_precomputed)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {time.time()-t0:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), args.save_path)
            print(f"[INFO] Saved best student -> {args.save_path}")
    print("Training finished. Best val acc: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-lr', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--save-path', type=str, default='best_student.pth')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--pretrain-teacher', action='store_true')
    parser.add_argument('--teacher-epochs', type=int, default=3)
    parser.add_argument('--teacher-lr', type=float, default=1e-3)
    parser.add_argument('--teacher-checkpoint', type=str, default='')
    parser.add_argument('--teacher-ckpt-path', type=str, default='teacher_quick_ckpt.pth')
    parser.add_argument('--num-classes', type=int, default=101)
    parser.add_argument('--precompute-teacher-logits', action='store_true', help='Precompute teacher logits and use them for distillation')
    parser.add_argument('--precompute-batch-size', type=int, default=64, help='batch size used when precomputing logits')
    args = parser.parse_args()

    if not args.pretrain_teacher and args.teacher_checkpoint == '':
        print("[WARN] Neither pretrain_teacher nor teacher_checkpoint provided; distillation may be poor.")
    main(args)
