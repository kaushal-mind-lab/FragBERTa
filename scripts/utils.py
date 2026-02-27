import math
import os

import safe
import yaml
from transformers import (
    RobertaTokenizerFast,
)


def time_elapsed(t1, t2):
    """Return time elapsed between t1 and t2 as 'X hours, Y minutes, Z seconds'."""
    elapsed = abs(t2 - t1)
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or len(parts) == 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return "time elapsed: " + ", ".join(parts)


# -----------------------------
# Get total training steps
# -----------------------------
def get_total_steps(num_gpus, N, B, A, E):
    steps_per_epoch = math.ceil(N / (B * num_gpus * A))
    total_steps = steps_per_epoch * E
    return total_steps


# -----------------------------
# Load config file from disk
# -----------------------------
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -----------------------------
# Convert SMILES to SAFE repr
# -----------------------------
def smiles_to_safe(smiles_str):
    return safe.encode(smiles_str, canonical=True, ignore_stereo=True)


# -----------------------------------------
# Convert SMILES to SAFE repr (more robust)
# -----------------------------------------
def safe_encode(smiles: str, slicer: str):
    try:
        return safe.encode(smiles, slicer=slicer, canonical=True, ignore_stereo=True)
    except Exception:
        return None


# -----------------------------
# Tokenization / preprocessing
# -----------------------------
def build_tokenizer(tokenizer_dir: str) -> RobertaTokenizerFast:
    tok = RobertaTokenizerFast.from_pretrained(tokenizer_dir, use_fast=True)
    # Ensure padding side and special tokens are sane for MLM
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    tok.padding_side = "right"
    return tok


# -----------------------------
# Tokenize
# -----------------------------
def tokenize_function(
    examples, tokenizer: RobertaTokenizerFast, max_len: int, col_name
):
    return tokenizer(
        examples[col_name],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_special_tokens_mask=True,
    )


# ---------------------------------
# Save config with appended params
# ---------------------------------
def save_updated_config(
    original_cfg: dict,
    output_dir: str,
    total_runtime: float,
    perplexity: float,
    total_params: int,
    trainable_params: int,
    train_steps: int,
):
    new_cfg = dict(original_cfg)

    # Add new keys
    new_cfg["TOTAL_RUNTIME"] = str(total_runtime)
    new_cfg["FINAL_EVAL_PERPLEXITY"] = float(perplexity)
    new_cfg["SAVED_AT"] = new_cfg["OUTPUT_DIR"]
    new_cfg["TOTAL_PARAMS"] = total_params
    new_cfg["TRAINABLE_PARAMS"] = trainable_params
    new_cfg["TRAINING STEPS"] = train_steps

    # Save updtaed keys in a new YAML file
    save_path = os.path.join(output_dir, "config_final.yaml")
    with open(save_path, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"[Config] Saved updated config to {save_path}")
