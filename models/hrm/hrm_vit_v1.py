from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HRViT_V1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HRViT_V1Carry:
    inner_carry: HRViT_V1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]

# ================================================================= #
# 请在 hr_vit_v1.py 文件顶部添加这个新的类
# ================================================================= #
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, image_size=32, patch_size=4, in_chans=3, embed_dim=512):
        super().__init__()
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用一个卷积层同时实现图像分块和线性嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入图像尺寸符合预期
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})."
        
        # x 的形状: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # -> (B, embed_dim, num_patches_sqrt, num_patches_sqrt)
        # -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x
# ================================================================= #

class HRViT_V1Config(BaseModel):
    # --- 视觉模型新增参数 ---
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10  # CIFAR-10 数据集有 10 个类别

    # --- 从原模型保留的参数 ---
    batch_size: int
    # seq_len: int # 我们将动态计算 seq_len，所以可以暂时注释或删除
    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

# ================================================================= #


class HRViT_V1Block(nn.Module):
    def __init__(self, config: HRViT_V1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HRViT_V1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HRViT_V1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HRViT_V1_Inner(nn.Module):
# ================================================================= #
# 请用以下代码替换 HRViT_V1_Inner 的 __init__ 方法
# ================================================================= #
    def __init__(self, config: HRViT_V1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        # --- 1. 视觉模型输入部分 ---
        self.patch_embed = PatchEmbed(
            image_size=config.image_size, patch_size=config.patch_size, 
            in_chans=config.in_chans, embed_dim=config.hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 为 [CLS] token 和所有 patch token 创建位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

        # --- 2. 视觉模型输出部分 (我们先定义好，后面再用) ---
        # 输出头的维度是 num_classes 而不是 vocab_size
        self.head = CastedLinear(self.config.hidden_size, self.config.num_classes, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # LM Blocks - 移除 RoPE, 使用可学习的位置编码
        assert self.config.pos_encodings == "learned", "HR-ViT only supports learned positional encodings."
        # 我们不再需要 rotary_emb 或 embed_pos

        # Reasoning Layers (这部分保持不变)
        self.H_level = HRViT_V1ReasoningModule(layers=[HRViT_V1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HRViT_V1ReasoningModule(layers=[HRViT_V1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states (这部分保持不变)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init (这部分保持不变)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    # ================================================================= #

    # ================================================================= #
    # 请用以下代码替换 HRViT_V1_Inner 的 _input_embeddings 方法
    # ================================================================= #
    def _input_embeddings(self, images: torch.Tensor):
        # images 的形状: (B, C, H, W)
        # 1. 通过 PatchEmbed 模块
        x = self.patch_embed(images)  # -> (B, num_patches, hidden_size)

        # 2. 准备并拼接 [CLS] token
        # 将 cls_token 从 (1, 1, D) 扩展到 (B, 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (B, num_patches + 1, hidden_size)

        # 3. 添加位置编码
        # pos_embed: (1, num_patches + 1, D) 会被自动广播到 (B, num_patches + 1, D)
        x = x + self.pos_embed
        
        # 注意：我们不再需要 embed_scale，因为 PatchEmbed 的卷积层和位置编码已经是可学习的了
        return x.to(self.forward_dtype)
    # ================================================================= #

    def empty_carry(self, batch_size: int):
        # 新的序列长度是 patch 数量 + 1个 [CLS] token
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        seq_len = num_patches + 1
        
        return HRViT_V1InnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRViT_V1InnerCarry):
        return HRViT_V1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

# ================================================================= #
# 请用以下代码替换 HRViT_V1_Inner 的 forward 方法
# ================================================================= #
    def forward(self, carry: HRViT_V1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HRViT_V1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 我们不再需要 RoPE，所以 seq_info 是空的
        seq_info = dict(
            cos_sin=None,
        )

        # 输入编码：从 batch 中获取 'images'
        # 注意：我们这里假设 batch 字典中有一个键叫 'images'
        input_embeddings = self._input_embeddings(batch["images"])

        # Forward iterations (这部分双循环的核心逻辑保持不变)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad (这部分也保持不变)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # --- 输出处理：使用新的分类头 ---
        new_carry = HRViT_V1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        # 提取 [CLS] token 对应的输出 (它在序列的第0个位置)
        cls_output = z_H[:, 0] 
        
        # 将 [CLS] token 的输出送入分类头，得到最终的 logits
        output_logits = self.head(cls_output)

        # Q head 的输入同样使用 [CLS] token 的状态
        q_logits = self.q_head(cls_output).to(torch.float32)
        
        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1])
    # ================================================================= #


class HRViT_V1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HRViT_V1Config(**config_dict)
        self.inner = HRViT_V1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    # ================================================================= #
    # 请用以下代码替换 HRViT_V1 的 initial_carry 方法
    # ================================================================= #
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["images"].shape[0] # 从 batch["images"] 获取 batch_size

        return HRViT_V1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            
            # 为 'images' 和 'labels' 初始化 current_data
            current_data={
                "images": torch.empty_like(batch["images"]),
                "labels": torch.empty_like(batch["labels"])
            }
        )
    # ================================================================= #
        
    # ================================================================= #
    # 请用以下代码替换 HRViT_V1 的 forward 方法
    # ================================================================= #
    def forward(self, carry: HRViT_V1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HRViT_V1Carry, Dict[str, torch.Tensor]]:
        # 更新数据, carry (移除已停止序列的旧数据，换上新数据)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # 这里的逻辑保持不变，但因为 current_data 的键已经更新，它会自动处理 'images' 和 'labels'
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # 调用内部模型 (这部分完全不变)
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # ACT 的停止逻辑 (这部分也完全不变)
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HRViT_V1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
    # ================================================================= #
