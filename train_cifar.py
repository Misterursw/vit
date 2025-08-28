# train_cifar.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import os
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
# 假设您的模型代码位于 'models' 文件夹下
# 我们需要从您修改后的 hrm_vit_v1.py 中导入模型
from models.hrm.hrm_vit_v1 import HRViT_V1, HRViT_V1Carry
# 导入 GradScaler 用于混合精度训练
from torch.cuda.amp import GradScaler, autocast

from models.hrm.hrm_vit_v1 import HRViT_V1, HRViT_V1Carry
# train_cifar.py (最终修正版，兼容您的 PyTorch 环境)
# 使用兼容性好的 API
from torch.cuda.amp import GradScaler, autocast

# 确保导入了正确的模型和损失头
from models.hrm.hrm_vit_v1 import HRViT_V1
# --- 关键修正：添加下面这行缺失的导入 ---
from models.losses import ACTLossHead
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import os
from tqdm import tqdm
from typing import Dict, Tuple

# 关键修正：导入旧版但兼容性好的 amp API
from torch.cuda.amp import GradScaler, autocast

from models.hrm.hrm_vit_v1 import HRViT_V1, HRViT_V1Carry

# --- EarlyStopping 类 (无需改动) ---
# --- 早停与检查点模块 ---
# --- 早停与智能检查点模块 (V4) ---
class CheckpointManager:
    def __init__(self, patience: int = 20, verbose: bool = True, path: str = 'checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        # 定义两个检查点路径
        self.best_path = path
        self.previous_best_path = os.path.join(os.path.dirname(path), "previous_" + os.path.basename(path))
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, state: Dict):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
            self.counter = 0

    def save_checkpoint(self, val_loss, state: Dict):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # --- 核心逻辑：轮换保存 ---
        # 1. 如果已存在 best_model，先将其重命名为 previous_best_model
        if os.path.exists(self.best_path):
            try:
                # 使用 shutil.move 来原子性地重命名
                shutil.move(self.best_path, self.previous_best_path)
                if self.verbose: print(f"Backed up previous best model to {self.previous_best_path}")
            except Exception as e:
                if self.verbose: print(f"Warning: Could not back up previous best model. Error: {e}")

        # 2. 保存新的 best_model
        torch.save(state, self.best_path)
        self.val_loss_min = val_loss

# --- 数据加载模块 (保持不变) ---
def get_dataloaders(data_path: str, batch_size: int, num_workers: int, augmentation_type: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    print(f"使用数据增强策略: {augmentation_type}")
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    
    train_transforms_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if augmentation_type == "autoaugment":
        train_transforms_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    
    train_transforms_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.RandomErasing(p=0.1)])
    
    transform_train = transforms.Compose(train_transforms_list)
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# --- 训练与验证 (逻辑微调) ---
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, scaler: GradScaler, act_loss_weight: float) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Training (ACT W: {act_loss_weight:.3f})", leave=False)
    
    # 获取内部的 HR-ViT 模型，用于创建 carry
    inner_model = model.module.model if isinstance(model, nn.DataParallel) else model.model

    for images, labels in progress_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch: Dict[str, torch.Tensor] = {"images": images, "labels": labels}
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            # --- 关键修正：恢复 while 循环 ---
            carry = inner_model.initial_carry(batch)
            while True:
                # model 是 ACTLossHead 包装器
                new_carry, loss, _, _, halted_all = model(
                    batch=batch,
                    carry=carry, # 传递 carry
                    return_keys=[],
                    act_loss_weight=act_loss_weight
                )
                carry = new_carry
                if halted_all:
                    break
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch: Dict[str, torch.Tensor] = {"images": images, "labels": labels}
            with autocast():
                # 为了验证，我们需要访问内部模型来获得logits
                inner_model_with_loss = model.module if isinstance(model, nn.DataParallel) else model
                inner_model = inner_model_with_loss.model
                
                # 创建初始状态时需要确保设备正确
                carry = inner_model.initial_carry(batch)
                while True:
                    carry, outputs = inner_model(carry=carry, batch=batch)
                    if carry.halted.all(): break
                logits = outputs['logits']
                loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return total_loss / len(loader), accuracy

# --- 主函数 (最终版) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Training Script for HR-ViT on CIFAR-100 (V4)")
    parser.add_argument('--config', type=str, default="config_cifar100.yaml", help="Path to the config file.")
    parser.add_argument('--resume', action='store_true', help="Resume training from the best_model checkpoint in the config path.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config["run"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(config["run"]["checkpoint_path"], exist_ok=True)
    checkpoint_file = os.path.join(config["run"]["checkpoint_path"], "best_model.pth")
    
    train_loader, val_loader = get_dataloaders(**config["training"])

    model_config = config["model"]
    model_config["batch_size"] = config["training"]["batch_size"]
    hr_vit_model = HRViT_V1(model_config)
    model = ACTLossHead(hr_vit_model, loss_type="softmax_cross_entropy")
    model.to(device)
    
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    start_epoch = 1
    act_warmup_activated = False
    act_warmup_start_epoch = -1

    manager = CheckpointManager(patience=config["training"]["early_stopping_patience"], verbose=True, path=checkpoint_file)

    if args.resume and os.path.isfile(checkpoint_file):
        print(f"加载检查点: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        manager.val_loss_min = checkpoint['best_val_loss']
        manager.best_score = -manager.val_loss_min
        act_warmup_activated = checkpoint.get('act_warmup_activated', False)
        act_warmup_start_epoch = checkpoint.get('act_warmup_start_epoch', -1)
        print(f"从 epoch {start_epoch} 继续。ACT 预热状态: {'已激活' if act_warmup_activated else '未激活'}")
    
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        act_config = config["training"]["act_loss"]
        act_loss_weight = 0.0
        if act_warmup_activated:
            progress = (epoch - act_warmup_start_epoch) / act_config["warmup_epochs"]
            act_loss_weight = min(1.0, progress)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, act_loss_weight)
        
        if epoch % config["training"]["validation_interval"] == 0:
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if not act_warmup_activated and val_acc >= act_config["trigger_accuracy_threshold"]:
                act_warmup_activated = True
                act_warmup_start_epoch = epoch
                print(f"--- 验证准确率达到 {val_acc:.2f}% (阈值 {act_config['trigger_accuracy_threshold']}%)! 开始预热 ACT Loss ---")
            
            current_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': min(val_loss, manager.val_loss_min),
                'act_warmup_activated': act_warmup_activated,
                'act_warmup_start_epoch': act_warmup_start_epoch,
            }
            manager(val_loss, current_state)
            if manager.early_stop:
                print("早停触发！")
                break
    
    print("\n训练结束。")
    print(f"性能最优的模型保存在: {checkpoint_file}")
    print(f"次优（前一最佳）模型保存在: {manager.previous_best_path}")