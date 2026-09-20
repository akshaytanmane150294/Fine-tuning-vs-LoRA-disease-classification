import torch
import os
from transformers import (AutoModelForCausalLM,
                           AutoConfig,
                           AutoTokenizer,
                           BitsAndBytesConfig,
                           )
from ClassificationHead import ClassificationHead
from DualBranchLoRA import (
    apply_dual_branch_lora,
    get_dual_branch_lora_state_dict,
    load_dual_branch_lora_state_dict,
)


class ClassificationNetLoRA(torch.nn.Module):
    def __init__(self, MODEL_NAME, DO_TEST, APPLY_LORA, NUM_CLASSES=24, r=8, lora_alpha=16):
        super(ClassificationNetLoRA, self).__init__()
        # Read Hugging Face token safely from environment or fallback
        token = os.environ.get("HF_TOKEN", None)
        model_name = MODEL_NAME
        compute_dtype = getattr(torch, "float16")

        # Load configuration from a pre-trained model with rope_scaling compatibility
        config = AutoConfig.from_pretrained(model_name, token=token)
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
                config.rope_scaling["type"] = config.rope_scaling["rope_type"]
            elif "type" in config.rope_scaling and "rope_type" not in config.rope_scaling:
                config.rope_scaling["rope_type"] = config.rope_scaling["type"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # Check CUDA availability for 4-bit QLoRA
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n[ERROR] CUDA GPU is not enabled or not detected!\n"
                "4-bit QLoRA LLM training requires a GPU.\n"
                "👉 In Google Colab: Go to 'Runtime' -> 'Change runtime type' -> Select 'T4 GPU' (or A100) -> Click 'Save', then re-run.\n"
            )

        self.model_name = MODEL_NAME
        self.APPLY_LORA = APPLY_LORA

        # Load pre-trained language model using native transformers support
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            token=token,
            trust_remote_code=False,
            device_map="auto",
            quantization_config=bnb_config,
        )

        # Replace language model head with an identity function
        self.llm.lm_head = torch.nn.Identity()

        if DO_TEST:
            if APPLY_LORA == True:
                apply_dual_branch_lora(self.llm, r=r, lora_alpha=lora_alpha)
                adapter_path = os.path.join('SavedAdapters', 'dual_branch_lora.pt')
                if os.path.exists(adapter_path):
                    state = torch.load(adapter_path, map_location="cpu")
                    load_dual_branch_lora_state_dict(self.llm, state)
                    print(f"[DualBranchLoRA] Loaded adapter weights from {adapter_path}")
            self.cls_head = ClassificationHead(config.hidden_size, num_classes=NUM_CLASSES)
            self.cls_head.load_state_dict(torch.load('SavedClassificationModels/clshead.pt'))
            self.cls_head.eval()
            return

        # ---------------- Apply Dual-Branch LoRA to LLM ----------------
        if APPLY_LORA == True:
            # Injects two parallel branches: (A -> B1) and (C -> B2)
            apply_dual_branch_lora(self.llm, r=r, lora_alpha=lora_alpha)
        else:
            # Freeze all parameters of the language model backbone
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        self.cls_head = ClassificationHead(config.hidden_size, num_classes=NUM_CLASSES)

    # forward pass
    def forward(self, input_ids, attention_mask):
        x = self.llm(input_ids, attention_mask).logits  # get last hidden state
        logits = self.cls_head(x)[:, -1, :]  # Apply classification head to the last token's output
        return logits

    def save_peft_adapter(self):
        if self.APPLY_LORA:
            os.makedirs('SavedAdapters', exist_ok=True)
            lora_state = get_dual_branch_lora_state_dict(self.llm)
            torch.save(lora_state, os.path.join('SavedAdapters', 'dual_branch_lora.pt'))
            print("[DualBranchLoRA] Saved dual-branch adapter weights to SavedAdapters/dual_branch_lora.pt")
        os.makedirs('SavedClassificationModels', exist_ok=True)
        torch.save(self.cls_head.state_dict(), os.path.join('SavedClassificationModels', 'clshead.pt'))
        print("[ClassificationHead] Saved head weights to SavedClassificationModels/clshead.pt")
