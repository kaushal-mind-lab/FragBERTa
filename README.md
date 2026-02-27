# FragBERTa

Fragment-aware transformer model using Sequential Attachement-based Fragment Encoding (SAFE) sequences for molecular representation learning and property prediction.

FragBERTa supports:
- Masked language model (MLM) pretraining
- Downstream finetuning with hyperparameter optimization
- SAFE-based prediction workflows

## Repository Structure

- `configs/` — training configuration files  
- `data/` — pretraining, finetuning, and tokenizer data  
- `models/` — trained model checkpoints  
- `scripts/` — training and evaluation scripts  
- `notebooks/` — exploratory experiments  

## Installation

### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then restart your terminal or run `source ~/.bashrc`.

### 2. Clone the repo and sync the environment
```bash
git clone https://github.com/kaushal-mind-lab/FragBERTa.git
cd FragBERTa
uv sync
```
This reads `pyproject.toml`, creates a `.venv` folder, and installs all dependencies including PyTorch with CUDA support. No conda required.

### 3. Activate the environment
```bash
source .venv/bin/activate
```

---

## Changing CUDA version

If your cluster has a different CUDA version, edit the index URL in `pyproject.toml`:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu121"  # change to cu118 or cu130 as needed
explicit = true
```

Then re-run `uv sync`.

---

## Running scripts

```bash
# With activation
source .venv/bin/activate
python scripts/pretrain/pretrain.py
```

---

## Adding a new package

```bash
uv add package-name
```

This installs the package, updates `pyproject.toml`, and regenerates `uv.lock` automatically.

---

Pretraining:
```bash
python scripts/pretrain.py --config configs/config_pretrain.yaml
```

Fine-tuning with hyperparameter optimization:
```bash
python scripts/finetuning_with_hpopt.py
```

Downstream prediction on SMILES:
```bash
python scripts/downstream_prediction_on_smiles.py
```

## License
See `LICENSE` file.