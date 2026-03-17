"""Quick validation: check MGRH config data paths resolve to real files."""

import yaml
from pathlib import Path

with open("configs/training/embedding_heads_bakeoff.yaml") as f:
    config = yaml.safe_load(f)

mgrh = config["mgrh_training"]
stages = mgrh["stages"]

print("=== Data path validation ===")
errors = []

for stage_name, stage_cfg in stages.items():
    root = Path(stage_cfg["data_root"])
    print(f"{stage_name}: data_root={root} (exists={root.exists()})")
    for src in stage_cfg.get("sources", []):
        p = root / src["path"]
        print(f"  {src['path']}: exists={p.exists()}, weight={src.get('weight','N/A')}")
        # Stage B/C sources should exist; Stage A won't (needs HF download)
        if stage_name != "stage_a" and not p.exists():
            errors.append(f"{stage_name} source missing: {p}")
    print()

# Base checkpoint
ckpt = Path(mgrh["base_checkpoint"])
print(f"Base checkpoint: {ckpt} (exists={ckpt.exists()})")
if not ckpt.exists():
    errors.append(f"Checkpoint missing: {ckpt}")

# Loss configs
print()
print("=== Loss config check ===")
for stage_name, stage_cfg in stages.items():
    loss = stage_cfg.get("loss", {})
    print(f"  {stage_name}: type={loss.get('type','N/A')}", end="")
    if "r_drop" in loss:
        print(f", r_drop.enabled={loss['r_drop']['enabled']}, alpha={loss['r_drop']['alpha']}", end="")
    if "contrastive_nli" in loss:
        print(f", contrastive_nli.enabled={loss['contrastive_nli']['enabled']}", end="")
    print()

# Eval config
print()
evl = mgrh["evaluation"]
print("=== Eval config check ===")
print(f"  selection_metric: {evl['selection_metric']}")
print(f"  gate_metrics: {evl['gate_metrics']}")
print(f"  eval_steps: {evl['eval_steps']}")
print(f"  save_steps: {evl['save_steps']}")
print(f"  ema.decay: {evl['ema']['decay']}")
print(f"  early_stopping.patience: {evl['early_stopping']['patience']}")

# ANCE config
ance = stages["stage_c"].get("ance", {})
print()
print("=== ANCE config ===")
print(f"  enabled: {ance.get('enabled')}")
print(f"  refresh_every_n_epochs: {ance.get('refresh_every_n_epochs')}")
print(f"  mine_top_k: {ance.get('mine_top_k')}")

# Calibration config
cal = stages["stage_c"].get("calibration", {})
print()
print("=== Calibration config ===")
print(f"  enabled: {cal.get('enabled')}")
print(f"  method: {cal.get('method')}")

print()
if errors:
    print(f"WARNINGS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
else:
    print("All checks passed.")
