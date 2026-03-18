"""Compute MaxSim population z-norm stats from benchmark data and update mgrh_metadata.json."""

import json
import time
from pathlib import Path

import torch
import numpy as np

CHECKPOINT = "D:/Modeling_studio/familyos_ultrabert/weights/pytorch"

print("Loading model...")
t0 = time.time()

from familyos_ultrabert.pytorch_inference import PyTorchInferenceEngine

engine = PyTorchInferenceEngine.load(CHECKPOINT, device="cuda")
print(f"Loaded in {time.time()-t0:.1f}s")

# Load benchmark pairs to compute population stats
listwise_path = Path("data/familyos/nli/relevance/human_benchmark_listwise.jsonl")

groups = []
with open(listwise_path, encoding="utf-8") as f:
    for line in f:
        groups.append(json.loads(line))

print(f"Computing MaxSim on {len(groups)} groups...")

head = engine.heads["relevance"]
all_maxsim = []

with torch.no_grad():
    for gi, group in enumerate(groups):
        query = group["query"]
        docs = [ep["text"] for ep in group["episodes"]]

        for doc in docs:
            # Encode
            joint = engine.tokenizer(
                query, doc, max_length=512, truncation=True,
                return_tensors="pt", padding=True,
            )
            enc_q = engine.tokenizer(
                query, max_length=512, truncation=True,
                return_tensors="pt", padding=True,
            )
            enc_d = engine.tokenizer(
                doc, max_length=512, truncation=True,
                return_tensors="pt", padding=True,
            )

            q_ids = enc_q["input_ids"].to(engine.device)
            q_mask = enc_q["attention_mask"].to(engine.device)
            d_ids = enc_d["input_ids"].to(engine.device)
            d_mask = enc_d["attention_mask"].to(engine.device)

            q_hidden = engine.encoder(
                input_ids=q_ids, attention_mask=q_mask, return_dict=True,
            ).last_hidden_state
            d_hidden = engine.encoder(
                input_ids=d_ids, attention_mask=d_mask, return_dict=True,
            ).last_hidden_state

            # Compute raw MaxSim (before z-norm)
            sim_matrix = torch.bmm(q_hidden, d_hidden.transpose(1, 2))
            sim_matrix = sim_matrix.masked_fill(~d_mask.unsqueeze(1).bool(), -1e9)
            maxsim_per_token = sim_matrix.max(dim=-1).values
            mask_float = q_mask.float()
            maxsim_val = (maxsim_per_token * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
            all_maxsim.append(maxsim_val.item())

        if (gi + 1) % 100 == 0:
            print(f"  {gi+1}/{len(groups)} groups done")

all_maxsim = np.array(all_maxsim)
pop_mean = float(all_maxsim.mean())
pop_std = float(all_maxsim.std())

print(f"\nMaxSim population stats (n={len(all_maxsim)}):")
print(f"  mean = {pop_mean:.6f}")
print(f"  std  = {pop_std:.6f}")

# Update mgrh_metadata.json
meta_path = Path(CHECKPOINT) / "mgrh_metadata.json"
with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)

meta["calibration"]["maxsim_population_mean"] = pop_mean
meta["calibration"]["maxsim_population_std"] = pop_std

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nUpdated {meta_path}")

# Also update the source checkpoint
source_meta = Path("outputs/final-nli/mgrh_metadata.json")
if source_meta.exists():
    with open(source_meta, encoding="utf-8") as f:
        src = json.load(f)
    src["calibration"]["maxsim_population_mean"] = pop_mean
    src["calibration"]["maxsim_population_std"] = pop_std
    with open(source_meta, "w", encoding="utf-8") as f:
        json.dump(src, f, indent=2)
    print(f"Updated {source_meta}")

print("Done.")
