"""
1. Generate predictions from a finetuned model.
2. Accepts a CSV of SMILES, converts them to SAFE format, then runs inference.
3. Run as: python downstream_prediction_on_smiles.py --target tox21 --model_path path/to/best_model --test_data path/to/test.csv --task_type reg --output_path predictions.csv
4. Input CSV must have a 'smiles' column. Any other columns are passed through to the output.
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from finetuning_with_hpopt import FragBERT_For_DownstreamTasks
from rdkit import Chem, RDLogger
from scipy.special import expit
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from utils import convert_smiles_to_safe

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
rng = random.Random(SEED)

OTHER_SLICERS = ["mmpa", "recap", "rotatable", "hr", "attach"]


def smiles_to_safe_parts(smiles: str):
    """
    Returns (slicer_name, safe_string_without_prefix) or (None, None) if all fail.
    """
    result = convert_smiles_to_safe(smiles, "brics")
    if result is not None:
        return "BRICS", result

    for slicer in OTHER_SLICERS:
        result = convert_smiles_to_safe(smiles, slicer)
        if result is not None:
            return slicer.upper(), result
    return None, None


def convert_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a df with a 'smiles' column, canonicalizes SMILES, converts to SAFE.
    Returns a new df with columns: input_smiles, canonical_smiles, slicer, safe, plus any
    other original columns. Rows that fail are dropped with a warning.
    """
    records = []
    failed_canon = 0
    failed_safe = 0

    print("Raw samples:", len(df))
    print(f"Converting {len(df)} SMILES to compatible SAFE format...")

    seen = set()
    for idx, row in df.iterrows():
        input_smi = str(row["smiles"]).strip()

        # Canonicalize
        try:
            mol = Chem.MolFromSmiles(input_smi)
            if mol is None:
                raise ValueError
            canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            print(f"Cannot canonicalize SMILES (row {idx}): {input_smi}")
            failed_canon += 1
            continue

        if canonical_smi in seen:
            continue
        seen.add(canonical_smi)

        # Convert to SAFE
        slicer_name, safe_str = smiles_to_safe_parts(canonical_smi)
        if safe_str is None:
            print(f"No SAFE encoding found (row {idx}): {input_smi}")
            failed_safe += 1
            continue

        records.append(
            {
                **{k: v for k, v in row.items() if k != "smiles"},
                "input_smiles": input_smi,
                "canonical_smiles": canonical_smi,
                "slicer": slicer_name,
                "safe": f"{slicer_name}.{safe_str}",  # prefixed — used as model input
                "safe_display": safe_str,  # without prefix — for output CSV
            }
        )

    total_failed = failed_canon + failed_safe
    print(f"Converted: {len(records)}/{len(df)}")
    if total_failed > 0:
        print(f"  Failed canonicalization: {failed_canon}, failed SAFE: {failed_safe}")
        print("  These rows will be excluded from predictions.")

    print("Processed samples:", len(seen))
    return pd.DataFrame(records).reset_index(drop=True)


# ========================================
# Dataset Class
# ========================================
class FragBERT_Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, num_labels=1):
        self.examples = tokenizer(
            text=texts,
            text_pair=None,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        # Dummy labels — required by Trainer.predict() but not used.
        # Shape must match model output: (N,) for reg/slclass, (N, num_labels) for mlclass.
        if num_labels > 1:
            self.labels = torch.zeros(len(texts), num_labels, dtype=torch.float)
        else:
            self.labels = torch.zeros(len(texts), dtype=torch.float)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        item["labels"] = self.labels[index]
        return item


# ========================================
# Main
# ========================================
def main():
    parser = argparse.ArgumentParser(description="FragBERT Inference")

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of task such as bace, tox21, sider, esol, etc.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model directory",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to CSV file with a 'smiles' column. Any other columns are passed through to output.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["reg", "slclass", "mlclass"],
        required=True,
        help="Task type the model was trained on",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="../data/tokenizer/roberta_fast_tokenizer_BPE",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=200,
        help="Max token length (must match training)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV",
    )

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    raw_df = pd.read_csv(args.test_data)
    if "smiles" not in raw_df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")
    print(f"{len(raw_df)} rows loaded")

    # Convert SMILES → SAFE (drops rows that fail)
    test_df = convert_smiles_column(raw_df)
    if len(test_df) == 0:
        raise RuntimeError(
            "No valid molecules remaining after SAFE conversion. Exiting."
        )

    texts = test_df["safe"].astype(str).tolist()
    print(f"  {len(texts)} molecules ready for inference")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.tokenizer_path, do_lower_case=False
    )

    # Build dataset
    if args.target == "tox21":
        num_labels = 12
    elif args.target == "sider":
        num_labels = 27
    else:
        num_labels = 1
    dataset = FragBERT_Dataset(texts, tokenizer, args.max_len, num_labels)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = FragBERT_For_DownstreamTasks.from_mlm_pretrained(
        pretrained_path=args.model_path,
        num_labels=num_labels,
        task_type=args.task_type,
    )

    # Use Trainer just for inference (no training)
    training_args = TrainingArguments(
        output_dir="/tmp/fragbert_predict",
        per_device_eval_batch_size=32,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(model=model, args=training_args)

    # Run predictions
    print("Running inference...")
    results = trainer.predict(dataset)
    raw_preds = results.predictions  # shape: (N, num_labels) or (N, 1)

    scaler_path = args.model_path+'/scaler.json'
    # Post-process predictions
    if args.task_type == "reg":
        preds = raw_preds.squeeze()
        if os.path.exists(scaler_path):
            print(f"Applying inverse scaling from {scaler_path}...")
            with open(scaler_path) as f:
                scaler_params = json.load(f)
            if scaler_params["type"] == "minmax":
                mn, mx = scaler_params["min"], scaler_params["max"]
                preds = preds * (mx - mn) + mn
            elif scaler_params["type"] == "standard":
                mean, std = scaler_params["mean"], scaler_params["std"]
                preds = preds * std + mean
        test_df["prediction"] = preds

    elif args.task_type == "slclass":
        probs = expit(raw_preds.squeeze())
        test_df["predicted_prob"] = probs
        test_df["predicted_class"] = (probs >= 0.5).astype(int)

    elif args.task_type == "mlclass":
        probs = expit(raw_preds)  # shape: (N, num_labels)
        pred_classes = (probs >= 0.5).astype(int)
        for i in range(probs.shape[1]):
            test_df[f"prob_label_{i}"] = probs[:, i]
            test_df[f"pred_label_{i}"] = pred_classes[:, i]

    # Reorder output: slicer | input_smiles | canonical_smiles | safe | other passthrough cols | predictions
    # Drop prefixed safe (model input), keep safe_display for output
    test_df = test_df.drop(columns=["safe"]).rename(columns={"safe_display": "safe"})
    fixed_cols = ["slicer", "input_smiles", "canonical_smiles", "safe"]
    pred_cols = [
        c
        for c in test_df.columns
        if c.startswith("pred")
        or c in ("prediction", "predicted_prob", "predicted_class")
        or c.startswith("prob_label_")
    ]
    passthrough_cols = [c for c in test_df.columns if c not in fixed_cols + pred_cols]
    test_df = test_df[fixed_cols + passthrough_cols + pred_cols]

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(test_df.head())


if __name__ == "__main__":
    main()
