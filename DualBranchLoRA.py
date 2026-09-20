import os
import math
import torch
import torch.nn as nn

class DualBranchLoRALinear(nn.Module):
    """
    Dual-Branch LoRA Layer:
    Implements two parallel low-rank adapter pathways alongside frozen base weights:
    Formula: h = Wx + (alpha / r) * [ (B1 * A)x + (B2 * C)x ]
    
    Branch 1: A (down-projection, N(0, sigma^2)) -> B1 (up-projection, zeros)
    Branch 2: C (down-projection, N(0, sigma^2)) -> B2 (up-projection, zeros)
    """
    def __init__(self, base_layer: nn.Module, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Determine in_features and out_features from base_layer
        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        
        if in_features is None or out_features is None:
            raise ValueError(f"Cannot determine in_features/out_features for {base_layer}")
            
        self.in_features = in_features
        self.out_features = out_features
        
        # Freeze original base layer weights
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Trainable LoRA parameters must be FP32 for PyTorch AMP GradScaler compatibility
        adapter_dtype = torch.float32
        device = None
        for p in self.base_layer.parameters():
            device = p.device
            break
            
        # ---------------- Branch 1: (A -> B1) ----------------
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=adapter_dtype)
        self.lora_B1 = nn.Linear(r, out_features, bias=False, device=device, dtype=adapter_dtype)
        
        # ---------------- Branch 2: (C -> B2) ----------------
        self.lora_C = nn.Linear(in_features, r, bias=False, device=device, dtype=adapter_dtype)
        self.lora_B2 = nn.Linear(r, out_features, bias=False, device=device, dtype=adapter_dtype)
        
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # Branch 1 Initialization: A is Kaiming/Gaussian, B1 is Zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B1.weight)
        
        # Branch 2 Initialization: C is Kaiming/Gaussian, B2 is Zeros
        nn.init.kaiming_uniform_(self.lora_C.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base Pretrained Output: W * x
        base_out = self.base_layer(x)
        
        # 2. Compute Adapter Paths
        target_dtype = self.lora_A.weight.dtype
        x_adapter = self.dropout(x.to(target_dtype))
        
        # Branch 1: (B1 * A) * x
        branch1_out = self.lora_B1(self.lora_A(x_adapter))
        
        # Branch 2: (B2 * C) * x
        branch2_out = self.lora_B2(self.lora_C(x_adapter))
        
        # 3. Sum: h = Wx + scaling * (Branch1 + Branch2)
        lora_out = self.scaling * (branch1_out + branch2_out)
        
        return base_out + lora_out.to(base_out.dtype)


def apply_dual_branch_lora(
    model: nn.Module, 
    r: int = 8, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.0,
    target_module_keywords=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
):
    """
    Recursively replaces target linear layers in the model with DualBranchLoRALinear.
    """
    # Freeze all parameters in the backbone model first
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0

    def _replace_modules(parent_module):
        nonlocal replaced_count
        for child_name, child_module in list(parent_module.named_children()):
            # Check if this child matches target keywords and has linear characteristics
            is_target = any(kw in child_name.lower() for kw in target_module_keywords)
            has_features = hasattr(child_module, "in_features") and hasattr(child_module, "out_features")
            
            if is_target and has_features and not isinstance(child_module, DualBranchLoRALinear):
                # Replace with DualBranchLoRALinear
                wrapped_layer = DualBranchLoRALinear(
                    base_layer=child_module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(parent_module, child_name, wrapped_layer)
                replaced_count += 1
            else:
                _replace_modules(child_module)

    _replace_modules(model)
    print(f"[DualBranchLoRA] Injected Dual-Branch LoRA into {replaced_count} linear layers.")
    return model


def get_dual_branch_lora_state_dict(model: nn.Module):
    """
    Extracts state_dict containing only trainable DualBranchLoRA parameters (lora_A, lora_B1, lora_C, lora_B2).
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, DualBranchLoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{name}.lora_B1.weight"] = module.lora_B1.weight.data.cpu()
            lora_state[f"{name}.lora_C.weight"] = module.lora_C.weight.data.cpu()
            lora_state[f"{name}.lora_B2.weight"] = module.lora_B2.weight.data.cpu()
    return lora_state


def load_dual_branch_lora_state_dict(model: nn.Module, state_dict: dict):
    """
    Loads saved DualBranchLoRA parameters into the model.
    """
    model_dict = model.state_dict()
    for key, val in state_dict.items():
        if key in model_dict:
            model_dict[key].copy_(val)
        else:
            print(f"[Warning] Key {key} not found in model during Dual-Branch LoRA loading.")
