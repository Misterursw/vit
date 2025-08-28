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


# models/hrm/hrm_vit_v1.py (真正最终修正版)

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
# 删除了 sparse_embedding 的导入，因为它在这个文件中没有被用到
# from models.sparse_embedding import CastedSparseEmbedding


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

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, image_size=32, patch_size=4, in_chans=3, embed_dim=512):
        super().__init__()
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class HRViT_V1Config(BaseModel, extra='allow'):
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10
    batch_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

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
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HRViT_V1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HRViT_V1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HRViT_V1_Inner(nn.Module):
    def __init__(self, config: HRViT_V1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.patch_embed = PatchEmbed(
            image_size=config.image_size, patch_size=config.patch_size, 
            in_chans=config.in_chans, embed_dim=config.hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.head = CastedLinear(self.config.hidden_size, self.config.num_classes, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        assert self.config.pos_encodings == "learned", "HR-ViT only supports learned positional encodings."

        self.H_level = HRViT_V1ReasoningModule(layers=[HRViT_V1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HRViT_V1ReasoningModule(layers=[HRViT_V1Block(self.config) for _i in range(self.config.L_layers)])
        
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size), std=1))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size), std=1))

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, images: torch.Tensor):
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x.to(self.forward_dtype)

    def empty_carry(self, batch_size: int, device: torch.device):
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        seq_len = num_patches + 1
        return HRViT_V1InnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, device=device, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len, self.config.hidden_size, device=device, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRViT_V1InnerCarry):
        # 此处的 .to(device) 是一个保险措施，核心修正已在其他地方完成
        H_init_on_device = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype)
        L_init_on_device = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype)
        return HRViT_V1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init_on_device, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init_on_device, carry.z_L),
        )

    def forward(self, carry: HRViT_V1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HRViT_V1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=None)
        input_embeddings = self._input_embeddings(batch["images"])

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        new_carry = HRViT_V1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        cls_output = z_H[:, 0]
        output_logits = self.head(cls_output)
        q_logits = self.q_head(cls_output).to(torch.float32)
        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1])


class HRViT_V1(nn.Module):
    """ACT wrapper."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HRViT_V1Config.model_validate(config_dict)
        self.inner = HRViT_V1_Inner(self.config)

    # 这是一个辅助属性，用来获取模型所在的设备
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["images"].shape[0]
        # 关键修正：获取模型当前设备
        device = self.device

        return HRViT_V1Carry(
            # 关键修正：将设备信息传递下去
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            # 关键修正：在创建时就指定正确的设备
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            # empty_like 会自动在与 batch['images'] 等相同的设备上创建
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HRViT_V1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HRViT_V1Carry, Dict[str, torch.Tensor]]:
        # 关键修正：确保 carry 中的张量在进入 reset_carry 之前是正确的设备
        # new_current_data 的创建逻辑已经可以保证这一点
        new_inner_carry = self.inner.reset_carry(carry.halted.to(self.device), carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}
        
        with torch.no_grad():
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
