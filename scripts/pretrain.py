# Run as CUDA_VISIBLE_DEVICES=0,1,2,3,4 OMP_NUM_THREADS=8 torchrun --nproc_per_node=5 --master_port=29500 pretrain.py &> pretrain.out

import os

os.environ["WANDB_MODE"] = "online"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_NCCL_TIMEOUT"] = "1800"  # 30 minutes
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import (
    build_tokenizer,
    get_total_steps,
    load_config,
    save_updated_config,
    time_elapsed,
    tokenize_function,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
t_start = time.time()


def save_mlm_checkpoint(trainer, tokenizer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(output_dir)  # 1. Save model + config
    tokenizer.save_pretrained(
        output_dir
    )  # 2. Save tokenizer (CRITICAL for SAFE + prefixes)
    trainer.state.save_to_json(
        output_dir / "trainer_state.json"
    )  # 3. Save training state (optional, but useful)
    print(f"MLM-pretrained model saved to: {output_dir}")


def main():
    # -----------------------------
    # Param reads
    # -----------------------------
    ORIG_CONFIG_PATH = "./config.yaml"
    cfg = load_config(ORIG_CONFIG_PATH)
    run_name = cfg["RUN_NAME"]
    data_csv = cfg["DATA_CSV"]
    tokenizer_dir = cfg["TOKENIZER_DIR"]
    output_dir = cfg["OUTPUT_DIR"]
    output_dir_best = cfg["OUTPUT_DIR_BEST"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir_best):
        os.makedirs(output_dir_best, exist_ok=True)

    hidden_size = int(cfg["HIDDEN_SIZE"])
    train_bs = int(cfg["TRAIN_BATCH_SIZE"])
    valid_bs = int(cfg["VALID_BATCH_SIZE"])
    epochs = int(cfg["TRAIN_EPOCHS"])
    lr = float(cfg["LEARNING_RATE"])
    weight_decay = float(cfg["WEIGHT_DECAY"])
    max_len = int(cfg["MAX_LEN"])
    max_pos_emb = int(cfg["MAX_POSITION_EMBEDDINGS"])
    num_heads = int(cfg["NUM_ATTENTION_HEADS"])
    num_layers = int(cfg["NUM_HIDDEN_LAYERS"])
    type_vocab = int(cfg["TYPE_VOCAB_SIZE"])
    N = int(cfg["NUM_DATAPOINTS"])
    train_col = cfg["TRAIN_COL"]
    val_fraction = float(cfg["VAL_FRACTION"])
    mlm_prob = float(cfg["MLM_PROBABILITY"])
    grad_accum = int(cfg.get("GRADIENT_ACCUMULATION_STEPS", 1))
    eval_accum = int(cfg.get("EVAL_ACCUMULATION_STEPS", 1))
    warmup_steps = int(cfg.get("WARMUP_STEPS", 0))
    fp16 = bool(cfg.get("FP16", True))
    bf16 = bool(cfg.get("BF16", False))
    logging_steps = int(cfg.get("LOGGING_STEPS", 50))
    save_strategy = cfg.get("SAVE_STRATEGY", "epoch")
    eval_strategy = cfg.get("EVAL_STRATEGY", "epoch")
    eval_steps = int(cfg["EVAL_STEPS"])
    save_total = int(cfg.get("SAVE_TOTAL_LIMIT", 3))
    load_best = bool(cfg.get("LOAD_BEST_MODEL_AT_END", True))
    orig_cfg_save_path = os.path.join(output_dir_best, "config_orig.yaml")
    resume_from_checkpoint = cfg.get("RESUME_FROM_CHECKPOINT", "auto")
    checkpoint_dir = None
    early_stopping_patience = int(cfg.get("EARLY_STOPPING_PATIENCE", 5))
    early_stopping_threshold = float(cfg.get("EARLY_STOPPING_THRESHOLD", 0.0))
    dataloader_n_workers = int(cfg.get("DATALOADER_NUM_WORKERS", 4))

    # -------------------------------------
    # Initialize Distributed Training FIRST
    # -------------------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = -1
        is_distributed = False

    if rank == 0:
        wandb.init(
            project=cfg["WANDB_PROJECT"],
            entity=cfg["WANDB_ENTITY"],
            name=cfg["WANDB_RUN_NAME"],
            config=cfg,
            resume="allow",
            id=cfg.get("WANDB_RUN_ID", None),
        )
    else:
        # Disable wandb completely on non-zero ranks
        os.environ["WANDB_DISABLED"] = "true"

    # --------------------------------------
    # Check if training needs to be resumed
    # --------------------------------------
    if rank == 0:
        if resume_from_checkpoint == "auto":
            checkpoints = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(output_dir, d))
            ]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                checkpoint_dir = os.path.join(output_dir, checkpoints[-1])
                print(f"Found checkpoint: {checkpoint_dir}")
            else:
                print("No checkpoint found, starting from scratch")
        elif resume_from_checkpoint and resume_from_checkpoint != "False":
            checkpoint_dir = resume_from_checkpoint
            print(f"Resuming from specified checkpoint: {checkpoint_dir}")

    # Broadcast checkpoint path to all ranks
    if is_distributed:
        checkpoint_list = [checkpoint_dir] if rank == 0 else [None]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint_dir = checkpoint_list[0]

    # ---------------------------------
    # Tokenization and Data Loading
    # ---------------------------------
    tokenizer = build_tokenizer(tokenizer_dir)
    vocab_size = len(tokenizer)

    # Create unique temp directory for this run
    temp_dataset_path = f"/tmp/tokenized_safe_{run_name}"

    if rank == 0:
        subprocess.run(["cp", ORIG_CONFIG_PATH, orig_cfg_save_path], check=True)
        print(f"Vocab size: {vocab_size}")

        if not os.path.exists(data_csv):
            raise FileNotFoundError(f"DATA_CSV not found: {data_csv}")

        columns_to_load = [train_col]
        df_raw = pd.read_parquet(data_csv)[columns_to_load]

        if N == 0:
            print("Using all samples")
            df = df_raw.copy()
        else:
            df = df_raw.sample(N, random_state=seed)
        del df_raw

        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Drop rows with missing values
        df = df.dropna()
        print(f"Samples after dropping NaN: {len(df)}")

        # Split train/validation
        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed, shuffle=True
        )

        # Create datasets
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

        ds_dict = DatasetDict(train=train_ds, validation=val_ds)
        # Tokenize
        tokenized_ds = ds_dict.map(
            lambda batch: tokenize_function(batch, tokenizer, max_len, train_col),
            batched=True,
            remove_columns=[train_col],
            desc="Tokenizing sequences",
        )
        tokenized_ds_memory = tokenized_ds
    else:
        tokenized_ds_memory = None

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Broadcast dataset object (as a Python pickled object)
    if is_distributed:
        obj_list = [tokenized_ds_memory]
        dist.broadcast_object_list(obj_list, src=0)
        tokenized_ds = obj_list[0]
    else:
        tokenized_ds = tokenized_ds_memory

    # -----------------------------
    # Data collation
    # -----------------------------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
    )

    # -----------------------------
    # Model configuration and setup
    # -----------------------------
    roberta_cfg = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=4 * hidden_size,
        max_position_embeddings=max_pos_emb,
        type_vocab_size=type_vocab,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=(
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.cls_token_id
        ),
        eos_token_id=(
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.sep_token_id
        ),
    )

    # Create multi-task model
    model = RobertaForMaskedLM(roberta_cfg)
    print("============================================================")
    print(model)
    print("============================================================")
    model.roberta.resize_token_embeddings(len(tokenizer))

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------------
    # Training setup
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=valid_bs,
        learning_rate=lr,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        eval_accumulation_steps=eval_accum,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total,
        logging_steps=logging_steps,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        fp16=fp16,
        bf16=bf16,
        prediction_loss_only=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        report_to="wandb" if rank == 0 else "none",
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=dataloader_n_workers,
        dataloader_pin_memory=True,
        local_rank=local_rank if is_distributed else -1,
    )

    if rank == 0:
        print("\n" + "=" * 50)
        print("MULTI-GPU TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"World size: {world_size}")
        print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        print(f"PyTorch sees: {torch.cuda.device_count()} GPU(s)")
        print(f"Trainer will use: {training_args.n_gpu} GPU(s)")
        print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
        print(
            f"Effective batch size: {training_args.per_device_train_batch_size * world_size * training_args.gradient_accumulation_steps}"
        )
        print(f"Parallel mode: {training_args.parallel_mode}")
        print("=" * 50 + "\n")

    # Create custom trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        ],
    )

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    N_samples = len(tokenized_ds["train"])
    B = train_bs
    A = grad_accum
    E = epochs

    train_steps = get_total_steps(num_gpus, N_samples, B, A, E)

    if rank == 0:
        print(f"Num GPUs: {num_gpus}")
        print(f"Total training steps: {train_steps}")

    # -----------------------------
    # Train
    # -----------------------------
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        if rank == 0:
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        if rank == 0:
            print("Starting training from scratch")
        trainer.train()

    # Synchronize all processes before final evaluation
    if is_distributed:
        dist.barrier()

    # -----------------------------
    # Evaluate
    # -----------------------------
    eval_metrics = trainer.evaluate()

    # -----------------------------
    # Saving (RANK 0 ONLY)
    # -----------------------------
    if rank == 0:
        print(f"[Eval] metrics: {eval_metrics}")

        ppl = (
            math.exp(eval_metrics["eval_loss"])
            if eval_metrics.get("eval_loss", None) is not None
            else float("nan")
        )
        print(f"[Eval] perplexity: {ppl:.4f}")

        # trainer.save_model(output_dir_best)
        # tokenizer.save_pretrained(output_dir_best)
        save_mlm_checkpoint(
            trainer=trainer, tokenizer=tokenizer, output_dir=output_dir_best
        )

        # Save metrics
        with open(os.path.join(output_dir, "training_eval_metrics.json"), "w") as f:
            json.dump({"eval": eval_metrics, "perplexity": ppl}, f, indent=2)

        print(f"[Saved] Model + tokenizer saved to {output_dir_best}")

        total_runtime = time_elapsed(time.time(), t_start)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        save_updated_config(
            cfg,
            output_dir_best,
            total_runtime,
            ppl,
            total_params,
            trainable_params,
            train_steps,
        )

    # Final barrier
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("Training complete!")

        # Clean up temp dataset
        if os.path.exists(temp_dataset_path):
            import shutil

            shutil.rmtree(temp_dataset_path)
            print(f"Cleaned up temp dataset: {temp_dataset_path}")


if __name__ == "__main__":
    main()
