import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from pydantic import BaseModel
from dataclasses import dataclass
import math
import traceback

# --- From common.py ---
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor

# --- From layers.py ---
CosSin = Tuple[torch.Tensor, torch.Tensor]

def _find_multiple(a, b):
    return (-(a // -b)) * b

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias=bias)

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # 为了通用性，这里使用PyTorch内置的注意力机制，因为flash_attn可能未安装
        attn_output = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=self.causal).transpose(1, 2)
        
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

# --- From hrm_vit_v1.py ---
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
    def __init__(self, image_size=32, patch_size=4, in_chans=3, embed_dim=512):
        super().__init__()
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, f"Input image size ({H}*{W}) doesn't match model."
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
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用建议的 Pre-Norm 结构以增强稳定性
        normed_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        hidden_states = hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=normed_states)
        normed_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        hidden_states = hidden_states + self.mlp(normed_states)
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
        self.patch_embed = PatchEmbed(image_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans, embed_dim=config.hidden_size)
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
    def empty_carry(self, batch_size: int):
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        seq_len = num_patches + 1
        return HRViT_V1InnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRViT_V1InnerCarry):
        H_init = self.H_init.to(dtype=carry.z_H.dtype)
        L_init = self.L_init.to(dtype=carry.z_L.dtype)
        return HRViT_V1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
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
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HRViT_V1Config(**config_dict)
        self.inner = HRViT_V1_Inner(self.config)
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["images"].shape[0]
        return HRViT_V1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={"images": torch.empty_like(batch["images"]), "labels": torch.empty_like(batch["labels"])}
        )
    def forward(self, carry: HRViT_V1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HRViT_V1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
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

# --- Test Script ---
if __name__ == '__main__':
    print("--- Starting Model Test ---")

    # 1. Model Configuration
    config = {
        "image_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 10,
        "batch_size": 4, # 增加batch size测试
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 4,
        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "learned",
        "halt_max_steps": 5,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32", # 在CPU上测试使用float32
    }

    # 2. Instantiate Model
    try:
        model = HRViT_V1(config)
        model.train() # 设置为训练模式以测试ACT逻辑
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        traceback.print_exc()
        exit()

    # 3. Create Dummy Data
    batch_size = config["batch_size"]
    in_chans = config["in_chans"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]

    dummy_images = torch.randn(batch_size, in_chans, image_size, image_size)
    # 尽管标签在核心模型前向传播中不直接使用，但ACT外壳需要它
    dummy_labels = torch.zeros((batch_size, 1), dtype=torch.long) # 标签形状可以简化

    dummy_batch = {
        "images": dummy_images,
        "labels": dummy_labels
    }
    print(f"Created dummy batch with image shape: {dummy_images.shape}")

    # 4. Perform a Forward Pass
    try:
        # 初始化 carry state
        initial_carry = model.initial_carry(dummy_batch)
        print("Initial carry state created.")

        # 运行模型
        new_carry, outputs = model(initial_carry, dummy_batch)
        print("Forward pass completed successfully.")

        # 5. Check Outputs
        output_logits = outputs['logits']
        print(f"\n--- Test Results ---")
        print(f"Output logits shape: {output_logits.shape}")
        print(f"Expected logits shape: {(batch_size, num_classes)}")

        assert output_logits.shape == (batch_size, num_classes)
        print("\nTest PASSED! The model ran successfully and the output dimensions are correct.")

    except Exception as e:
        print(f"\n--- Test FAILED ---")
        print(f"An error occurred during the forward pass: {e}")
        traceback.print_exc()