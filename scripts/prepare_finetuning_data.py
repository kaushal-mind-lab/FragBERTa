import os
import torch
import numpy as np
import pandas as pd
import random
import json
from utils import canonicalize_smiles, brics_or_random_slicer
import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = random.Random(seed)

targnames = ['bace', 'bbbp', 'tox21', 'hiv', 'sider', 'esol', 'freesolv', 'lipo', 'pdbbind']
other_slicers = ["mmpa", "recap", "rotatable", "hr", "attach"]
input_root = "/data/kaushal/FragBERTa/data/finetune/interim"
output_root = "/data/kaushal/FragBERTa/data/finetune/cleaned"
os.makedirs(output_root, exist_ok=True)
datasets = {}

for targname in targnames:
    print("================================")
    print("Processing target:", targname)

    input_path = f"{input_root}/{targname}.csv"
    output_path = f"{output_root}/{targname}_cleaned.csv"
    
    df = pd.read_csv(input_path)
    ini_num_samples = len(df)
    print("Rows in raw file:", ini_num_samples)
    
    df['canonical_smiles'] = df['smiles'].apply(lambda x: canonicalize_smiles(x.strip()))
    df = df.drop(columns=['smiles'])
    df = df.rename(columns={'canonical_smiles':'smiles'})
    df = df.dropna(subset=['smiles'])
    print("Rows after filtering invalid SMILES:", len(df))

    if len(df[df["smiles"].duplicated(keep=False)])>0:
        if targname in ['esol', 'freesolv', 'lipo', 'pdbbind']:
            col = [c for c in df.columns if c != "smiles"][0]
            df_avg = (df.groupby("smiles", as_index=False, sort=False)[col].mean())
            df = df_avg.copy(deep=True)
            print("Rows after averaging samples with multiple regression values:", len(df))
        elif targname in ['bace', 'bbbp', 'hiv']:
            col = [c for c in df.columns if c != "smiles"][0]
            df_consistent = (df.groupby("smiles", sort=False).filter(lambda x: x[col].nunique() == 1).drop_duplicates("smiles"))
            df = df_consistent.copy(deep=True)
            print("Rows after removing samples with opposing binary targets:", len(df))
        else:
            #TODO: Handle mult-label datasets
            print("Out of my scope at the moment. Performing")
            #df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
            #print("Rows after deduplicating canonical SMILES:", len(df))
            

    df_smiles = df.loc[:,'smiles'].values
    assert len(np.unique(df_smiles))==len(df), "Something's up"
    df_targ = df.drop(columns=["smiles"]).to_numpy() #df.iloc[:,1:].values
    
    rows_processed = []
    smiles_unprocessed = 0
    for smi, targ in zip(df_smiles, df_targ):
        safe_list = brics_or_random_slicer(smi, other_slicers, rng)
        if len(safe_list) == 0:
            print("NO SAFE FOUND FOR CANONICALIZED SMILES:", smi)
            smiles_unprocessed += 1
            continue
        for safe_str in safe_list:
            rows_processed.append((smi, safe_str, *np.atleast_1d(targ)))

    print("SMILES not converted to SAFE:", smiles_unprocessed)
    datasets[targname] = (ini_num_samples), len(rows_processed)

    if targname in ['sider', 'tox21']:
        columns = ['smiles', 'safe'] + [f'targ_{i}' for i in range(df_targ.shape[1])]
    else:
        columns = ['smiles', 'safe', 'target']

    out_df = pd.DataFrame(rows_processed, columns=columns)
    out_df.to_csv(output_path, encoding="utf8", index=False, header=True)

datasets = {"dataset": ('original_samples', 'valid_samples'), **datasets}
with open(f"{output_root}/cleaning_stats.json", "w") as fh:
    json.dump(datasets, fh, indent=4)