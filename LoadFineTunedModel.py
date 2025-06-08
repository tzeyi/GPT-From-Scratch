from pathlib import Path
from GPTModel import GPTModel
import torch


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# Configurations
finetuned_model_path = Path("gpt2-medium355M-sft.pth")
if not finetuned_model_path.exists():
    print(
        f"Could not find '{finetuned_model_path}'.\n"
        "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
    )

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
model = GPTModel(BASE_CONFIG)

# Load in model weights
model.load_state_dict(torch.load(
    "gpt2-medium355M-sft.pth",
    map_location=torch.device("mps"),
    weights_only=True
))
model.eval()
