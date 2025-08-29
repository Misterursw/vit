# models/hrm/hrm_vit_b_hybrid.py

import torch
import torch.nn as nn
import timm
from typing import List
from collections import OrderedDict

# 假设您的 HRViT_V1Block 和 HRViT_V1ReasoningModule 与 hrm_vit_v1.py 中的定义一致
# 我们直接从 hrm_vit_v1.py 中导入它们
from .hrm_vit_v1 import HRViT_V1Block, HRViT_V1ReasoningModule, HRViT_V1Config

class HRViT_B_Pure_Core(nn.Module):
    """
    一个使用ViT-B词嵌入层，但核心完全由H-Layers和L-Layers构成的模型。
    它完全移除了ViT原生的12个Transformer Block。
    """
    def __init__(self, hrm_config: HRViT_V1Config, num_classes=100, pretrained=True):
        super().__init__()
        self.hrm_config = hrm_config
        self.forward_dtype = getattr(torch, hrm_config.forward_dtype)

        # 1. 加载临时ViT-B模型以获取词嵌入层
        vit_b_temp = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        embed_dim = vit_b_temp.embed_dim

        # 2. 复制ViT的词嵌入部分
        self.patch_embed = vit_b_temp.patch_embed
        self.cls_token = vit_b_temp.cls_token
        self.pos_embed = vit_b_temp.pos_embed
        self.pos_drop = vit_b_temp.pos_drop
        
        # 3. 冻结词嵌入部分的参数
        print("Freezing Patch Embedding, CLS token, and Position Embedding.")
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

        # 4. 创建全新的H-Level和L-Level核心
        #    根据您的描述，我们创建6个H-Layer和6个L-Layer
        self.H_level = HRViT_V1ReasoningModule(
            layers=[HRViT_V1Block(self.hrm_config) for _ in range(self.hrm_config.H_layers)]
        )
        self.L_level = HRViT_V1ReasoningModule(
            layers=[HRViT_V1Block(self.hrm_config) for _ in range(self.hrm_config.L_layers)]
        )

        # 5. 初始化H和L状态
        self.H_init = nn.Parameter(torch.zeros(1, 1, embed_dim, dtype=self.forward_dtype))
        self.L_init = nn.Parameter(torch.zeros(1, 1, embed_dim, dtype=self.forward_dtype))
        nn.init.normal_(self.H_init, std=1e-6)
        nn.init.normal_(self.L_init, std=1e-6)
        
        # 6. 创建最终的分类头
        self.norm = nn.LayerNorm(embed_dim) 
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 释放临时模型
        del vit_b_temp

    def _pos_embed(self, x):
        """处理CLS Token和位置编码"""
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=self.forward_dtype):
            # 1. 通过ViT的词嵌入层
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            
            # 此时x是经过编码的patch序列，准备进入全新的Transformer核心

            # 2. 初始化 z_H 和 z_L 状态
            batch_size, seq_len, embed_dim = x.shape
            z_H = self.H_init.expand(batch_size, seq_len, embed_dim)
            z_L = self.L_init.expand(batch_size, seq_len, embed_dim)

            # 3. 通过您的H/L推理模块进行循环推理
            for _ in range(self.hrm_config.H_cycles):
                for _ in range(self.hrm_config.L_cycles):
                    # 输入注入：将词嵌入层的输出作为原始输入注入到L-Level
                    z_L = self.L_level(hidden_states=z_L, input_injection=(z_H + x), cos_sin=None)
                # L-Level的输出注入到H-Level
                z_H = self.H_level(hidden_states=z_H, input_injection=z_L, cos_sin=None)

            # 4. 使用最终的z_H状态进行分类
            cls_token_output = z_H[:, 0]
        
        # 5. 通过分类头
        x = self.norm(cls_token_output.to(torch.float32))
        x = self.head(x)
        
        return x

# --- 如何使用与对比 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    print("\n" + "="*50)
    print("Standard ViT-B Model Architecture (For Reference)")
    print("="*50)
    standard_vit_b = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    print(standard_vit_b)

    config = HRViT_V1Config(
        batch_size=4, hidden_size=768, num_heads=12, expansion=4.0,
        pos_encodings="learned", H_cycles=2, L_cycles=2, H_layers=6, L_layers=6,
        image_size=224, patch_size=16, in_chans=3, num_classes=100,
        halt_max_steps=16, halt_exploration_prob=0.1, forward_dtype="bfloat16" 
    )

    print("\n" + "="*50)
    print("Pure H/L-Core ViT-B Model Architecture")
    print("="*50)
    pure_core_model = HRViT_B_Pure_Core(hrm_config=config, num_classes=100, pretrained=True)
    print(pure_core_model)
    pure_core_model.to(device)

    print("\n" + "="*80)
    print("Detailed Parameter Status in Pure H/L-Core Model")
    print("="*80)
    for name, param in pure_core_model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"{name:<60} | Status: {status}")
    print("="*80)

    print("\nRunning a dummy forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    output = pure_core_model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Dummy forward pass successful.")