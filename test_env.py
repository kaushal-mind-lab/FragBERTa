import os
import time
import wandb
import json
import math
import shutil
import subprocess
import random
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, roc_auc_score
from scipy.special import softmax, expit

import torch
import torch.distributed as dist
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.data import Dataset
import chemprop

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from dataclasses import dataclass
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    Trainer,
    set_seed,
    TrainingArguments,
    RobertaTokenizerFast,
    #EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

import warnings
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars"
)