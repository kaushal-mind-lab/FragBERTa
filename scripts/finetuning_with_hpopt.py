"""
1. Hyperparameter optimization for FragBERT finetuning via Optuna
2. Training model from scratch on train set using the best-fit params
3. Run as: python finetuning_with_hpopt.py --target tox21 --task_type mlclass --use_scaffold 0 --pretrained_model_path path/to/pretrained/model --optuna_num_trials 150
"""
#TODO: Remove emojis from prints statements
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#,1,2,3,4,5,6,7"
os.environ["WANDB_MODE"] = "disabled"
import argparse
import json
import random
import shutil
import warnings
from pathlib import Path

import chemprop
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.special import expit, softmax
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import Dataset
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    RobertaTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)

my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)


# ========================================
# Model Class
# ========================================
class FragBERT_For_DownstreamTasks(RobertaPreTrainedModel):
    def __init__(self, config, task_type="reg"):
        super().__init__(config)
        assert task_type in {
            "reg",  # "regression",
            "slclass",  # "single_label_classification",
            "mlclass",  # "multi_label_classification",
        }, f"Invalid task_type: {task_type}"

        self.task_type = task_type
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:

            if self.task_type == "reg":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.task_type == "slclass":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(), labels.float().squeeze())
            elif self.task_type == "mlclass":
                valid_mask = ~torch.isnan(labels)  # Handle NaN in labels
                if valid_mask.any():

                    labels_masked = torch.where(
                        valid_mask, labels, torch.zeros_like(labels)
                    )  # Replace NaN with 0 (will be masked out)
                    loss_fct = BCEWithLogitsLoss(
                        reduction="none"
                    )  # Compute per-element loss
                    loss_per_element = loss_fct(logits, labels_masked.float())
                    loss_per_element = (
                        loss_per_element * valid_mask.float()
                    )  # Mask out NaN positions
                    loss = (
                        loss_per_element.sum() / valid_mask.float().sum()
                    )  # Average only over valid elements
                else:
                    loss = torch.tensor(0.0, device=logits.device)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_mlm_pretrained(
        cls,
        pretrained_path: str,
        num_labels: int,
        task_type: str,
    ):
        """
        Load MLM-pretrained checkpoint and attach downstream head.
        """

        # Load config and update for regression
        config = RobertaConfig.from_pretrained(pretrained_path)
        config.num_labels = num_labels
        # print(f"Config loaded:")
        # print(f"  Vocab size: {config.vocab_size}")
        # print(f"  Hidden size: {config.hidden_size}")
        # print(f"  Num layers: {config.num_hidden_layers}")
        # print(f"  Num labels: {config.num_labels}")
        model = cls.from_pretrained(
            pretrained_path,
            config=config,
            ignore_mismatched_sizes=True,  # safely drops MLM head
            task_type=task_type,
        )
        return model


# ========================================
# Dataset Class
# ========================================
class FragBERT_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        text, labels = data
        self.examples = tokenizer(
            text=text,
            text_pair=None,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        item["labels"] = self.labels[index]
        return item


# ========================================
# Data Preparation Functions
# ========================================
def prepare_data(config, use_cache=True):
    """Prepare and cache data splits to avoid reprocessing during HPO"""
    cache_dir = Path(config["output_dir"]) / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_files = {
        "train": cache_dir / "train_df.csv",
        "val": cache_dir / "val_df.csv",
        "test": cache_dir / "test_df.csv",
    }

    # Check if cache exists
    if use_cache and all(f.exists() for f in cache_files.values()):
        print("✓ Loading cached data splits...")
        train_df = pd.read_csv(cache_files["train"])
        val_df = pd.read_csv(cache_files["val"])
        test_df = pd.read_csv(cache_files["test"])
        return train_df, val_df, test_df

    print("Processing data from scratch...")
    main_df = pd.read_csv(config["dataset"])

    if config["target"] in ["sider", "tox21"]:
        print("\n" + "=" * 60)
        print("MULTI-LABEL DATASET - Preserving NaN for masking")
        print("=" * 60)

        target_cols = [col for col in main_df.columns if col not in ["safe", "smiles"]]
        nan_counts = main_df[target_cols].isna().sum()
        total_nan = nan_counts.sum()
        print(f"Found {total_nan} missing values (will be masked during training)")
        non_zero_nan = nan_counts[nan_counts > 0]
        print("Missing values per label:")
        for col, count in non_zero_nan.items():
            print(f"  {col}: {count}")
        print("=" * 60 + "\n")

        for colname in target_cols:
            main_df[colname] = pd.to_numeric(main_df[colname], errors="coerce")

    else:
        main_df["target"] = pd.to_numeric(main_df["target"], errors="raise")

    print(f"Original dataset size: {len(main_df)}")
    main_df = main_df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    print(f"After deduplication: {len(main_df)}")

    # Validate SMILES before scaffold split to avoid segfault
    if config["use_scaffold"] == 1:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings

        print("Validating SMILES strings...")
        valid_mask = main_df["smiles"].apply(
            lambda x: Chem.MolFromSmiles(str(x)) is not None
        )
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            print(f"⚠ Found {invalid_count} invalid SMILES, removing them...")
            main_df = main_df[valid_mask].reset_index(drop=True)
            print(f"Dataset size after cleaning: {len(main_df)}")
    # =========================================
    print(f"Original dataset size (after SMILES validation): {len(main_df)}")

    # Split data
    if config["use_scaffold"] == 0:
        print("Performing random splits")
        train_df, val_df = train_test_split(main_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

        if config["target"] not in ["sider", "tox21"]:
            colnames = ["safe", "target"]
        else:
            colnames = ["safe"] + [f"targ_{i}" for i in range(main_df.shape[1] - 2)]

        train_df = train_df[colnames]
        val_df = val_df[colnames]
        test_df = test_df[colnames]

    else:
        # Scaffold split implementation (simplified)
        print("Performing scaffold splits")
        molecule_list = [
            chemprop.data.data.MoleculeDatapoint(smiles=[row["smiles"]], targets=None)
            for _, row in main_df.iterrows()
        ]
        molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)
        train, val, test = chemprop.data.scaffold.scaffold_split(
            data=molecule_dataset, sizes=(0.8, 0.1, 0.1), seed=42, balanced=True
        )

        train2match = pd.DataFrame({"smiles": np.array(train.smiles()).flatten()})
        val2match = pd.DataFrame({"smiles": np.array(val.smiles()).flatten()})
        test2match = pd.DataFrame({"smiles": np.array(test.smiles()).flatten()})
        #print(len(main_df),len(train2match), len(val2match), len(test2match))

        merged_df_train = pd.merge(main_df, train2match, on="smiles", how="inner")
        merged_df_val = pd.merge(main_df, val2match, on="smiles", how="inner")
        merged_df_test = pd.merge(main_df, test2match, on="smiles", how="inner")
        #print(len(main_df),len(merged_df_train), len(merged_df_val), len(merged_df_test))

        if config["target"] not in ["sider", "tox21"]:
            colnames = ["safe", "target"]
        else:
            colnames = ["safe"] + [f"targ_{i}" for i in range(main_df.shape[1] - 2)]

        train_df = merged_df_train[colnames]
        val_df = merged_df_val[colnames]
        test_df = merged_df_test[colnames]

    print(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Cache the splits
    train_df.to_csv(cache_files["train"], index=False)
    val_df.to_csv(cache_files["val"], index=False)
    test_df.to_csv(cache_files["test"], index=False)
    print("✓ Data cached for future use")

    return train_df, val_df, test_df


# ========================================
# Metrics computation
# ========================================
def make_compute_metrics(task_type="reg", scaler=None):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if task_type == "mlclass":
            preds = np.asarray(preds)  # Keep all columns
            labels = np.asarray(labels)
        else:
            predictions = [i[0] for i in preds]
            preds = np.asarray(predictions)
            labels = np.asarray(labels)

        # -------------------------
        # Regression
        # -------------------------
        if task_type == "reg":
            preds = preds.reshape(-1, 1) if preds.ndim == 1 else preds
            labels = labels.reshape(-1, 1) if labels.ndim == 1 else labels

            if scaler is not None:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)

            mse = mean_squared_error(y_true=labels, y_pred=preds)
            rmse = root_mean_squared_error(y_true=labels, y_pred=preds)
            mae = mean_absolute_error(y_true=labels, y_pred=preds)

            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
            }

        # -------------------------
        # Single-label classification
        # -------------------------
        elif task_type == "slclass":

            # Case 1: single-logit binary classification
            if preds.ndim == 1 or preds.shape[1] == 1:
                probs = expit(preds.reshape(-1))
                auc = roc_auc_score(labels.reshape(-1), probs)

            # Case 2: two-logit softmax classification
            else:
                probs = softmax(preds, axis=1)
                auc = roc_auc_score(labels, probs[:, 1])

            return {"aucroc": auc}

        # -------------------------
        # Multi-label classification
        # -------------------------
        elif task_type == "mlclass":
            # preds: (N, C), labels: (N, C)
            probs = expit(preds)  # ---or use torch.sigmoid after preds.detach().numpy()
            valid_mask = ~np.isnan(labels)
            try:
                auc_scores = []
                for label_idx in range(labels.shape[1]):
                    label_mask = valid_mask[:, label_idx]  # Get mask for this label
                    if label_mask.sum() > 0:
                        label_true = labels[label_mask, label_idx]
                        label_probs = probs[label_mask, label_idx]
                        if (
                            len(np.unique(label_true)) > 1
                        ):  # Check if both classes are present
                            auc = roc_auc_score(label_true, label_probs)
                            auc_scores.append(auc)
                if len(auc_scores) > 0:  # Average over labels with valid AUC
                    AUC = np.mean(auc_scores)
                else:
                    AUC = 0.0
            except ValueError as e:
                print(f"Warning: Could not compute test ROC-AUC - {e}")
                AUC = 0.0

            return {"aucroc": AUC}

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    return compute_metrics


# ========================================
# Optuna Callback for Pruning
# ========================================
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, monitor="eval_rmse"):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Report intermediate value and check for pruning
        if self.monitor in metrics:
            self.trial.report(metrics[self.monitor], state.epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


# ========================================
# Objective Function for Optuna
# ========================================
def objective(trial, config, train_df, val_df, tokenizer):
    """Objective function for hyperparameter optimization"""

    scaler_choice = (
        trial.suggest_categorical("scaler", [0, 1, 2])
        if config["task_type"] == "reg"
        else trial.suggest_categorical("scaler", [0])
    )  # 0=none, 1=minmax, 2=standard

    # Sample hyperparameters
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "train_batch_size": trial.suggest_categorical(
            "train_batch_size", [8, 16, 32, 64]
        ),
        "freeze_strategy": trial.suggest_categorical("freeze_strategy", [0, 1, 2]),
        "num_freeze_layers": (
            trial.suggest_int("num_freeze_layers", 0, 7)
            if trial.params.get("freeze_strategy") == 2
            else 0
        ),
        "scaler": scaler_choice,
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]
        ),
    }

    trial_output_dir = Path(config["output_dir"]) / f"trial_{trial.number}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Prepare data with scaling
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        if hyperparams["scaler"] == 0:
            scaler = None
        elif hyperparams["scaler"] == 1:
            scaler = MinMaxScaler()
            train_scaled["target"] = scaler.fit_transform(train_scaled[["target"]])
            val_scaled["target"] = scaler.transform(val_scaled[["target"]])
        elif hyperparams["scaler"] == 2:
            scaler = StandardScaler()
            train_scaled["target"] = scaler.fit_transform(train_scaled[["target"]])
            val_scaled["target"] = scaler.transform(val_scaled[["target"]])

        # Create datasets
        safe_col = train_scaled["safe"].astype(str).tolist()
        target_cols = train_scaled.drop(columns=["safe"]).values.squeeze().tolist()
        train_data = (safe_col, target_cols)
        train_dataset = FragBERT_Dataset(train_data, tokenizer, config["max_len"])

        val_safe_col = val_scaled["safe"].astype(str).tolist()
        val_target_cols = val_scaled.drop(columns=["safe"]).values.squeeze().tolist()
        val_data = (val_safe_col, val_target_cols)
        val_dataset = FragBERT_Dataset(val_data, tokenizer, config["max_len"])

        # Load model
        model = FragBERT_For_DownstreamTasks.from_mlm_pretrained(
            pretrained_path=config["pretrained_model"],
            num_labels=config["num_labels"],
            task_type=config["task_type"],
        )

        # Apply freezing strategy
        if hyperparams["freeze_strategy"] == 1:
            for param in model.roberta.parameters():
                param.requires_grad = False
        elif hyperparams["freeze_strategy"] == 2:
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
            num_layers = len(model.roberta.encoder.layer)
            k = min(hyperparams["num_freeze_layers"], num_layers)
            for i in range(k):
                for param in model.roberta.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(trial_output_dir),
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=config["num_epochs_hpo"],
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            per_device_train_batch_size=hyperparams["train_batch_size"],
            per_device_eval_batch_size=config["validation_batch_size"],
            gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
            logging_strategy="epoch",
            report_to="none",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=(
                "rmse" if config["task_type"] == "reg" else "aucroc"
            ),
            greater_is_better=(config["task_type"] != "reg"),
            disable_tqdm=True,
        )

        # Trainer with pruning callback
        monitor_metric = "eval_rmse" if config["task_type"] == "reg" else "eval_aucroc"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=make_compute_metrics(
                task_type=config["task_type"], scaler=scaler
            ),
            callbacks=[OptunaPruningCallback(trial, monitor=monitor_metric)],
        )

        # Train
        print(f"\n{'='*25}")
        print("TRYING NEW HYPERPARAM SET")
        print(f"{'='*25}")

        trainer.train()
        eval_metrics = trainer.evaluate()

        # Select objective metric based on task
        if config["task_type"] == "reg":
            objective_value = eval_metrics["eval_rmse"]
        else:
            objective_value = eval_metrics["eval_aucroc"]

        # Cleanup
        shutil.rmtree(trial_output_dir, ignore_errors=True)

        return objective_value

    except optuna.TrialPruned:
        shutil.rmtree(trial_output_dir, ignore_errors=True)
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        shutil.rmtree(trial_output_dir, ignore_errors=True)
        raise


# ========================================
# Final Training with Best Hyperparameters
# ========================================
def train_final_model(best_params, config, train_df, val_df, test_df, tokenizer):
    """Train final model with best hyperparameters on full epochs"""

    print(f"\n{'='*60}")
    print("FINAL TRAINING WITH BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    print(json.dumps(best_params, indent=2))

    final_output_dir = Path(config["output_dir"]) / "final_model"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data with best scaler
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    if best_params["scaler"] == 0:
        scaler = None
    elif best_params["scaler"] == 1:
        scaler = MinMaxScaler()
        train_scaled["target"] = scaler.fit_transform(train_scaled[["target"]])
        scaler_params = {"type": "minmax", "min": float(scaler.data_min_[0]), "max": float(scaler.data_max_[0])}
        val_scaled["target"] = scaler.transform(val_scaled[["target"]])
        test_scaled["target"] = scaler.transform(test_scaled[["target"]])
    elif best_params["scaler"] == 2:
        scaler = StandardScaler()
        train_scaled["target"] = scaler.fit_transform(train_scaled[["target"]])
        scaler_params = {"type": "standard", "mean": float(scaler.mean_[0]), "std": float(scaler.scale_[0])}
        val_scaled["target"] = scaler.transform(val_scaled[["target"]])
        test_scaled["target"] = scaler.transform(test_scaled[["target"]])

    # Create datasets
    train_safe_col = train_scaled["safe"].astype(str).tolist()
    train_target_cols = train_scaled.drop(columns=["safe"]).values.squeeze().tolist()
    train_data = (train_safe_col, train_target_cols)
    train_dataset = FragBERT_Dataset(train_data, tokenizer, config["max_len"])

    val_safe_col = val_scaled["safe"].astype(str).tolist()
    val_target_cols = val_scaled.drop(columns=["safe"]).values.squeeze().tolist()
    val_data = (val_safe_col, val_target_cols)
    val_dataset = FragBERT_Dataset(val_data, tokenizer, config["max_len"])

    test_safe_col = test_scaled["safe"].astype(str).tolist()
    test_target_cols = test_scaled.drop(columns=["safe"]).values.squeeze().tolist()
    test_data = (test_safe_col, test_target_cols)
    test_dataset = FragBERT_Dataset(test_data, tokenizer, config["max_len"])

    # Load model
    model = FragBERT_For_DownstreamTasks.from_mlm_pretrained(
        pretrained_path=config["pretrained_model"],
        num_labels=config["num_labels"],
        task_type=config["task_type"],
    )

    # Apply best freezing strategy
    if best_params["freeze_strategy"] == 1:
        print("✓ Freezing entire encoder")
        for param in model.roberta.parameters():
            param.requires_grad = False
    elif best_params["freeze_strategy"] == 2:
        print(f"✓ Freezing first {best_params['num_freeze_layers']} layers")
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        num_layers = len(model.roberta.encoder.layer)
        k = min(best_params["num_freeze_layers"], num_layers)
        for i in range(k):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    # Training arguments with best hyperparameters
    training_args = TrainingArguments(
        output_dir=str(final_output_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config["num_epochs_final"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        per_device_train_batch_size=best_params["train_batch_size"],
        per_device_eval_batch_size=config["validation_batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        logging_strategy="epoch",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=("rmse" if config["task_type"] == "reg" else "aucroc"),
        greater_is_better=(config["task_type"] != "reg"),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(
            task_type=config["task_type"], scaler=scaler
        ),
    )

    # Final Train after hpopt
    print("\nStarting final training...")
    trainer.train()

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS")
    print("=" * 60)
    val_metrics = trainer.evaluate()
    print(json.dumps(val_metrics, indent=2))

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    test_results = trainer.predict(test_dataset)
    compute_metrics_fn = make_compute_metrics(
        task_type=config["task_type"], scaler=scaler
    )
    test_metrics = compute_metrics_fn(
        (test_results.predictions, test_results.label_ids)
    )
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    print(json.dumps(test_metrics, indent=2))

    # Save best model
    best_model_dir = Path(config["output_dir"]) / "best_model"
    trainer.save_model(str(best_model_dir))
    if scaler is not None:
        with open(f"{best_model_dir}/scaler.json", "w") as f:
                json.dump(scaler_params, f)
    print(f"\n✓ Best model saved to {best_model_dir}")
    # Save results
    results = {
        "best_hyperparameters": best_params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(Path(config["output_dir"]) / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ========================================
# Main Execution
# ========================================
def main():

    parser = argparse.ArgumentParser(description="FragBERT Hyperparameter Optimization")

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target property (e.g., esol, lipophilicity, bbbp)",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        choices=["reg", "slclass", "mlclass"],
        required=True,
        help="Task type",
    )

    parser.add_argument(
        "--use_scaffold",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use scaffold split (1) or random split (0)",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/kaushal/FragBERTa",
        required=True,
        help="Root path",
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path of the pretrained model checkpoint you want to finetune",
    )

    parser.add_argument(
        "--optuna_num_trials",
        type=int,
        default=150,
        required=True,
        help="Path of the pretrained model checkpoint you want to finetune",
    )

    args = parser.parse_args()

    target = args.target
    task_type = args.task_type
    if task_type == "reg":
        dataset_type = "regression"
    else:
        dataset_type = "classification"
    use_scaffold = args.use_scaffold
    if use_scaffold == 0:
        split_type = "random_split"
    elif use_scaffold == 1:
        split_type = "scaffold_split"
    if task_type in ["reg", "slclass"]:
        num_labels = 1
    else:
        if target == "sider":
            num_labels = 27
        elif target == "tox21":
            num_labels = 12

    # ==========================================================
    # CONFIGURATION - YOU AS USER SHOULD MODIFY THIS AS NEEDED
    # ==========================================================
    CONFIG = {
        
        # Paths
        "target": target,
        "dataset_type": dataset_type,
        "task_type": task_type,  # ---scaler=None for classification
        "num_labels": num_labels,  # 1 for reg and slclass, and 3 for mlclass
        "pretrained_model": args.pretrained_model_path,  # ---path to pretrained model
        "tokenizer": f"{args.root_dir}/data/tokenizer/roberta_fast_tokenizer_BPE",
        "dataset": f"{args.root_dir}/data/finetune/cleaned/{target}_cleaned.csv",
        "output_dir": f"{args.root_dir}/models/finetuned/{target}_on_{split_type}/hpopt_results",
        
        # Dataset parameters
        "use_scaffold": use_scaffold,  # 0=random, 1=scaffold
        "max_len": 200,
        
        # Optuna parameters
        "n_trials": args.optuna_num_trials,  # Number of hyperparameter combinations to try
        "n_jobs": 1,  # Parallel trials (set to 1 for single GPU)
        "timeout": None,  # Optional: timeout in seconds
        
        # Fixed hyperparameters (not optimized)
        "validation_batch_size": 16,
        "num_epochs_hpo": 40,  # Epochs per trial (shorter for HPO)
        "num_epochs_final": 150,  # Epochs for final training with best params
    }

    print(f"\n{'='*60}")
    print("FRAGBERT REGRESSION HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Prepare data (cached after first run)
    print("Preparing data...")
    train_df, val_df, test_df = prepare_data(CONFIG)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        CONFIG["tokenizer"], do_lower_case=False
    )

    # Create Optuna study
    print(f"\nStarting hyperparameter optimization with {CONFIG['n_trials']} trials...")
    study = optuna.create_study(
        direction=(
            "minimize" if CONFIG["task_type"] == "reg" else "maximize"
        ),  # Minimize RMSE or maximize aucroc
        sampler=TPESampler(seed=my_seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="fragbert_regression_hpo",
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, CONFIG, train_df, val_df, tokenizer),
        n_trials=CONFIG["n_trials"],
        n_jobs=CONFIG["n_jobs"],
        timeout=CONFIG["timeout"],
        show_progress_bar=True,
    )

    # Print optimization results
    print(f"\n{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    if CONFIG["task_type"] == "reg":
        print(f"Best RMSE: {study.best_value:.4f}")
    else:
        print(f"Best AUROC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    # Save study results
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)

    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    # Train final model with best hyperparameters
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}\n")

    final_results = train_final_model(
        study.best_params, CONFIG, train_df, val_df, test_df, tokenizer
    )

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print(f"Best model saved to: {CONFIG['output_dir']}/best_model")
    if CONFIG["task_type"] == "reg":
        print(f"\nFinal Test RMSE: {final_results['test_metrics']['test_rmse']:.4f}")
    else:
        print(
            f"\nFinal Test AUCROC: {final_results['test_metrics']['test_aucroc']:.4f}"
        )


if __name__ == "__main__":
    main()
