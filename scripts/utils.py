import os
import math
import yaml
from rdkit import Chem
import safe
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

def get_total_steps(num_gpus, N, B, A, E):
    """Get total training steps on multi-GPU training"""
    steps_per_epoch = math.ceil(N / (B * num_gpus * A))
    total_steps = steps_per_epoch * E
    return total_steps

def load_config(path: str) -> dict:
    """
    Load config file from disk
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# def convert_smiles_to_safe(smiles_str):
#     return safe.encode(smiles_str, canonical=True, ignore_stereo=True)

def convert_smiles_to_safe(smiles: str, slicer: str="brics"):
    """
    Convert SMILES to SAFE repr
    """
    try:
        return safe.encode(smiles, slicer=slicer, canonical=True, ignore_stereo=True)
    except Exception:
        return None

def canonicalize_smiles(smiles: str):
    """
    Canonicalize a SMILES using rdkit
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Cannot canonicalize SMILES:', smiles)
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def build_tokenizer(tokenizer_dir: str) -> RobertaTokenizerFast:
    """
    convert tokenizer to roberta fast tokenizer
    """
    tok = RobertaTokenizerFast.from_pretrained(tokenizer_dir, use_fast=True)
    if tok.pad_token is None: # Ensure padding side and special tokens are sane for MLM
        tok.add_special_tokens({"pad_token": "<pad>"})
    tok.padding_side = "right"
    return tok

def tokenize_function(examples,
                      tokenizer: RobertaTokenizerFast,
                      max_len: int,
                      col_name
):
    """
    Tokenize corpus
    """
    return tokenizer(
        examples[col_name],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_special_tokens_mask=True,
    )


def save_updated_config(
    original_cfg: dict,
    output_dir: str,
    total_runtime: float,
    perplexity: float,
    total_params: int,
    trainable_params: int,
    train_steps: int,
):
    """
    Save config with appended params
    """
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


def brics_or_random_slicer(smiles, other_slicers, rng):
    """
    Returns list of SAFE strings:
      - includes BRICS SAFE if available
      - includes one randomly chosen SAFE from other slicers if BRICS is not available
      - returns [] if all fail
    Each SAFE is prefixed with slicer tag.
    """
    out = []

    brics = convert_smiles_to_safe(smiles, "brics")
    if brics is not None:
        out.append(f"BRICS.{brics}")
        return out
    else:
        # Try other slicers; collect successes
        candidates = []
        for s in other_slicers:
            x = convert_smiles_to_safe(smiles, s)
            if x is not None:
                candidates.append((s, x))

        if len(candidates) > 0:
            s, x = candidates[rng.randrange(len(candidates))]
            out.append(f"{s.upper()}.{x}")
        
        return out