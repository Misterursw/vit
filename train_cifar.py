# train_cifar.py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import os
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
# 假设您的模型代码位于 'models' 文件夹下
# 我们需要从您修改后的 hrm_vit_v1.py 中导入模型
from models.hrm.hrm_vit_v1 import HRViT_V1, HRViT_V1Carry
# 导入 GradScaler 用于混合精度训练
from torch.cuda.amp import GradScaler, autocast
import shutil

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
from collections import OrderedDict
# 关键修正：导入旧版但兼容性好的 amp API
from torch.cuda.amp import GradScaler, autocast

from models.hrm.hrm_vit_v1 import HRViT_V1, HRViT_V1Carry
# --- 新增：加载部分预训练权重的函数 ---
# --- 修正后的权重加载函数 ---
# ... (CheckpointManager, get_dataloaders, get_linear_schedule_with_warmup 函数保持不变)
class CheckpointManager:
    def __init__(self, patience: int = 20, verbose: bool = True, path: str = 'checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
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
        if os.path.exists(self.best_path):
            try:
                shutil.move(self.best_path, self.previous_best_path)
                if self.verbose: print(f"Backed up previous best model to {self.previous_best_path}")
            except Exception as e:
                if self.verbose: print(f"Warning: Could not back up previous best model. Error: {e}")
        torch.save(state, self.best_path)
        self.val_loss_min = val_loss

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

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# --- 带有调试打印的权重加载函数 ---
# --- 最终修正的权重加载函数 ---
def load_partial_pretrained_weights(model: nn.Module, pretrained_path: str, device: torch.device):
    if not os.path.exists(pretrained_path):
        print(f"警告: 预训练权重文件不存在于 {pretrained_path}。将从零开始训练。")
        return

    print(f"正在从 {pretrained_path} 加载预训练权重...")
    checkpoint_data = torch.load(pretrained_path, map_location=device)
    
    if 'model_state_dict' in checkpoint_data:
        pretrained_state_dict = checkpoint_data['model_state_dict']
    elif 'state_dict' in checkpoint_data:
        pretrained_state_dict = checkpoint_data['state_dict']
    else:
        pretrained_state_dict = checkpoint_data

    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    loaded_keys = []
    skipped_keys = []

    for k, v in pretrained_state_dict.items():
        # --- 核心修正：移除 "_orig_mod." 前缀 ---
        if k.startswith("_orig_mod."):
            # “扒掉”外包装，得到真正的层名
            new_key = k[len("_orig_mod."):]
        else:
            new_key = k
        
        # 用处理后的层名进行匹配
        if new_key in model_state_dict and model_state_dict[new_key].shape == v.shape:
            new_state_dict[new_key] = v
            loaded_keys.append(new_key)
        else:
            skipped_keys.append(k) # 记录原始的、未加载的key
    
    # 使用 strict=False 安全地加载匹配上的权重
    model.load_state_dict(new_state_dict, strict=False)

    print(f"成功加载了 {len(loaded_keys)} 个层的权重。")
    if skipped_keys:
        print(f"跳过了 {len(skipped_keys)} 个不匹配的层 (这是正常的)。")

# --- train_one_epoch 和 validate_one_epoch 函数保持不变 ---
from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp import GradScaler, autocast

# train_cifar.py (请替换 train_one_epoch 函数)

# train_cifar.py (请使用这个版本的 train_one_epoch 函数)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, scaler: GradScaler, act_loss_weight: float, scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
    model.train()
    total_loss = 0.0
    total_ponder_steps = 0.0
    inner_model = model.module.model if isinstance(model, nn.DataParallel) else model.model
    
    batch_bar = tqdm(loader, desc=f"Epoch Training (ACT W: {act_loss_weight:.3f})")

    for batch_idx, (images, labels) in enumerate(batch_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch: Dict[str, torch.Tensor] = {"images": images, "labels": labels}
        optimizer.zero_grad(set_to_none=True)
        
        loss = None
        
        with autocast():
            carry = inner_model.initial_carry(batch)
            halt_max_steps = inner_model.config.halt_max_steps
            ponder_step = 0

            while True:
                ponder_step += 1
                new_carry, current_loss, _, _, halted_all = model(
                    batch=batch,
                    carry=carry,
                    return_keys=[],
                    act_loss_weight=act_loss_weight
                )
                
                loss = current_loss
                carry = new_carry
                
                if halted_all or ponder_step >= halt_max_steps:
                    break
        
        total_ponder_steps += ponder_step

        if loss is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
            avg_ponder = total_ponder_steps / (batch_idx + 1)
            batch_bar.set_postfix(
                loss=f"{loss.item():.3f}", 
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
                avg_steps=f"{avg_ponder:.2f}/{halt_max_steps}"
            )
        
    return total_loss / len(loader)
# train_cifar.py (请替换整个 validate_one_epoch 函数)

def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_ponder_steps = 0.0 # 新增：用于统计验证时的平均思考步数
    inner_model_with_loss = model.module if isinstance(model, nn.DataParallel) else model
    inner_model = inner_model_with_loss.model
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch: Dict[str, torch.Tensor] = {"images": images, "labels": labels}
            
            with autocast():
                carry = inner_model.initial_carry(batch)
                halt_max_steps = inner_model.config.halt_max_steps
                outputs = None
                ponder_step = 0

                # --- [核心修正] ---
                # 验证时也使用循环，并根据 q_head 决策来提前停止
                while True:
                    ponder_step += 1
                    # 在评估模式下，我们直接调用模型的核心逻辑 HRViT_V1
                    new_carry, current_outputs = inner_model(carry=carry, batch=batch)
                    
                    # 将当前步骤的输出保存在 outputs 中
                    # 注意：我们只保存尚未停机样本的 logits
                    # 第一次循环时，outputs 为 None
                    if outputs is None:
                        outputs = current_outputs
                    else:
                        # 用新一步的输出，更新那些仍在“思考”的样本的输出
                        is_still_running = ~carry.halted 
                        outputs["logits"] = torch.where(is_still_running.view(-1, 1), current_outputs["logits"], outputs["logits"])

                    carry = new_carry
                    
                    # 验证时的停机条件：所有样本都已停机，或达到最大步数
                    if carry.halted.all() or ponder_step >= halt_max_steps:
                        break
            
            total_ponder_steps += ponder_step
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_ponder_val = total_ponder_steps / len(loader)
    print(f"Validation Avg Steps: {avg_ponder_val:.2f}/{inner_model.config.halt_max_steps}")
    return total_loss / len(loader), accuracy
# --- 主函数 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Training Script for HR-ViT on CIFAR-100")
    parser.add_argument('--config', type=str, default="config_cifar100.yaml", help="Path to the config file.")
    parser.add_argument('--resume', action='store_true', help="Resume training from the best_model checkpoint.")
    parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging.")
    parser.add_argument('--pretrained-weights', type=str, default=None, help="Path to a pretrained model checkpoint for transfer learning.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_config = config["wandb"]
        if wandb_config.get("dir"):
            os.makedirs(wandb_config["dir"], exist_ok=True)
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            config=config,
            dir=wandb_config.get("dir")
        )

    device = torch.device(config["run"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(config["run"]["checkpoint_path"], exist_ok=True)
    checkpoint_file = os.path.join(config["run"]["checkpoint_path"], "best_model.pth")

    train_loader, val_loader = get_dataloaders(**config["training"])

    model_config = config["model"]
    # 确保 batch_size 参数传递给模型配置
    model_config["batch_size"] = config["training"]["batch_size"]
    hr_vit_model = HRViT_V1(model_config)
    model = ACTLossHead(hr_vit_model, loss_type="softmax_cross_entropy")
    model.to(device)

    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.pretrained_weights and not args.resume:
        print(f"--- 正在加载预训练权重: {args.pretrained_weights} ---")
        load_partial_pretrained_weights(model, args.pretrained_weights, device)

    optimizer_config = config["training"]
    betas = (optimizer_config.get("beta1", 0.9), optimizer_config.get("beta2", 0.95))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        betas=betas
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    num_training_steps = config["training"]["epochs"] * len(train_loader)
    num_warmup_steps = config["training"]["warmup_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    start_epoch = 1
    act_warmup_activated = False
    act_warmup_start_epoch = -1
    manager = CheckpointManager(patience=config["training"]["early_stopping_patience"], verbose=True, path=checkpoint_file)

    if args.resume and os.path.isfile(checkpoint_file):
        print(f"--- 正在从检查点恢复训练: {checkpoint_file} ---")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # 安全加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 建议在改变 halt_max_steps 后不加载优化器状态，但如果配置未变，可以加载
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        manager.val_loss_min = checkpoint.get('best_val_loss', float('inf'))
        manager.best_score = -manager.val_loss_min if manager.val_loss_min != float('inf') else None
        act_warmup_activated = checkpoint.get('act_warmup_activated', False)
        act_warmup_start_epoch = checkpoint.get('act_warmup_start_epoch', -1)
        print(f"--- 恢复成功，将从 Epoch {start_epoch} 开始 ---")

    # --- [核心修改] 将 epoch 变量传入 train_one_epoch ---
    def train_one_epoch_with_epoch_logging(epoch, model, loader, optimizer, device, scaler, act_loss_weight, scheduler):
        model.train()
        total_loss = 0.0
        total_ponder_steps = 0.0
        inner_model = model.module.model if isinstance(model, nn.DataParallel) else model.model

        # --- [核心修改] 在进度条描述中加入 Epoch 信息 ---
        batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']} Training (ACT W: {act_loss_weight:.3f})")

        for batch_idx, (images, labels) in enumerate(batch_bar):
            # ... (内部循环逻辑保持不变) ...
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch = {"images": images, "labels": labels}
            optimizer.zero_grad(set_to_none=True)
            loss = None
            with autocast():
                carry = inner_model.initial_carry(batch)
                halt_max_steps = inner_model.config.halt_max_steps
                ponder_step = 0
                while True:
                    ponder_step += 1
                    new_carry, current_loss, _, _, halted_all = model(
                        batch=batch, carry=carry, return_keys=[], act_loss_weight=act_loss_weight)
                    loss = current_loss
                    carry = new_carry
                    if halted_all or ponder_step >= halt_max_steps:
                        break
            total_ponder_steps += ponder_step
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # 注意：学习率调度器应该在每个 step 更新，而不是每个 epoch
                scheduler.step()
                total_loss += loss.item()
                avg_ponder = total_ponder_steps / (batch_idx + 1)
                batch_bar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    avg_steps=f"{avg_ponder:.2f}/{halt_max_steps}"
                )
        return total_loss / len(loader)

    # --- 主训练循环 ---
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        # --- [核心修改] 在每个 Epoch 开始时打印日志 ---
        print(f"\n{'='*20} Starting Epoch {epoch}/{config['training']['epochs']} {'='*20}")

        act_config = config["training"]["act_loss"]
        act_loss_weight = 0.0
        if act_warmup_activated:
            progress = (epoch - act_warmup_start_epoch) / act_config["warmup_epochs"]
            act_loss_weight = min(1.0, progress)
            print(f"ACT Loss is active. Weight: {act_loss_weight:.4f}")

        # --- [核心修改] 调用新的带日志的训练函数 ---
        train_loss = train_one_epoch_with_epoch_logging(epoch, model, train_loader, optimizer, device, scaler, act_loss_weight, scheduler)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "act_loss_weight": act_loss_weight
            })

        # --- 验证逻辑 ---
        if epoch % config["training"]["validation_interval"] == 0:
            print(f"\n--- Running Validation for Epoch {epoch} ---")
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            print(f"--- Epoch {epoch} Validation Result | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ---")

            if use_wandb:
                wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "epoch": epoch})

            if not act_warmup_activated and val_acc >= act_config["trigger_accuracy_threshold"]:
                act_warmup_activated = True
                act_warmup_start_epoch = epoch
                print(f"--- !!! ACT Loss Triggered !!! Accuracy {val_acc:.2f}% reached threshold {act_config['trigger_accuracy_threshold']}% ---")

            current_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': min(val_loss, manager.val_loss_min),
                'act_warmup_activated': act_warmup_activated,
                'act_warmup_start_epoch': act_warmup_start_epoch,
            }
            manager(val_loss, current_state)
            if manager.early_stop:
                print("--- Early stopping triggered. Ending training. ---")
                break

    print("\nTraining finished.")
    print(f"Best model saved to: {checkpoint_file}")
    if use_wandb:
        wandb.finish()