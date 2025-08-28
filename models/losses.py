from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100

# softmax_cross_entropy 函数保持不变，因为它是标准的
def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        # 对于分类任务，我们直接使用 PyTorch 的标准交叉熵损失
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str],
        act_loss_weight: float = 1.0,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # 这一步调用的是 HRViT_V1.forward()
        new_carry, outputs = self.model(carry=carry, batch=batch)
        labels = new_carry.current_data["labels"]

        # --- 关键修正：为图像分类重新定义“正确性” ---
        with torch.no_grad():
            # 对于图像分类，"正确性"就是一个形状为 (batch_size,) 的布尔张量
            # is_correct 的每个元素代表对应图片的预测是否正确
            is_correct = (torch.argmax(outputs["logits"], dim=-1) == labels)

            # Metrics (只统计已停机样本)
            valid_metrics = new_carry.halted
            metrics = {
                "count": valid_metrics.sum(),
                # 在已停机样本中，正确预测的数量
                "accuracy": (valid_metrics & is_correct).sum(),
                # 对于分类任务，"exact_accuracy" 和 "accuracy" 是一样的
                "exact_accuracy": (valid_metrics & is_correct).sum(),
                # q_halt_logits > 0 意为模型倾向于停机。此指标衡量停机决策的正确性
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == is_correct)).sum(),
                # 计算已停机样本的平均思考步数
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # --- 关键修正：为图像分类重新定义损失计算 ---
        
        # 1. 分类损失 (Classification Loss)
        # 使用标准的交叉熵，并对整个批次求和
        lm_loss = self.criterion(outputs["logits"], labels)

        # 2. Q-Halt 损失
        # 现在的目标 is_correct (B,) 和输入 q_halt_logits (B,) 形状完全匹配！
        batch_size = labels.shape[0]
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum"
        ) / batch_size # <--- 归一化

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # 3. Q-Continue 损失 (逻辑不变)
        # 3. Q-Continue 损失 (同样进行归一化)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            ) / batch_size # <--- 归一化

        # 将动态权重应用于 ACT 相关的损失
        total_loss = lm_loss + act_loss_weight * 0.5 * (q_halt_loss + q_continue_loss)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        halted_all = new_carry.halted.all()

        # 返回训练循环需要的所有值
        return new_carry, total_loss, metrics, detached_outputs, halted_all
