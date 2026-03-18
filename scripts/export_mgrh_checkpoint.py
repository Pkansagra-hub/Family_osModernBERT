"""
Export MGRH Checkpoint — Merge trained MGRH head into full deployable checkpoint.

Takes:
  - base_checkpoint: Full model from stage_b/best-ema (encoder + all frozen heads)
  - mgrh_checkpoint: stage_c/best/ directory (mgrh_head.pt + pair_encoder.pt)

Produces a checkpoint in the same format as stage_b/best-ema:
  model.safetensors  — encoder + all heads (including mgrh under key "heads.relevance.*")
  config.json        — model config
  tokenizer.*        — tokenizer files
  capabilities.json  — updated to include "relevance" capability
  embedding_metadata.json — preserved from base
  mgrh_metadata.json — new: records MGRH architecture + source checkpoint

The mgrh_head.pt at best/ was saved from EMA (head_to_save = ema_model.module),
so no extra EMA unwrapping is needed — we load it directly.

Usage:
    python scripts/export_mgrh_checkpoint.py \
        --base   outputs/embedding-bakeoff/stage_b/best-ema \
        --mgrh   outputs/embedding-bakeoff/mgrh/stage_c/best \
        --output outputs/embedding-bakeoff/mgrh/final-merged \
        [--config configs/training/embedding_heads_bakeoff.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml_config(config_path: Path) -> dict:
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        import json as _json
        with open(config_path, encoding="utf-8") as f:
            return _json.load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Merge MGRH head into full deployable checkpoint")
    parser.add_argument(
        "--base", required=True,
        help="Path to base checkpoint (stage_b/best-ema or equivalent)",
    )
    parser.add_argument(
        "--mgrh", required=True,
        help="Path to MGRH checkpoint directory containing mgrh_head.pt",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for merged checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/training/embedding_heads_bakeoff.yaml",
        help="Path to YAML config for MGRH head architecture params",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate files exist and print plan without writing output",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    mgrh_path = Path(args.mgrh)
    output_path = Path(args.output)
    config_path = Path(args.config)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    errors = []
    for required in [
        base_path / "model.safetensors",
        base_path / "config.json",
        base_path / "capabilities.json",
        mgrh_path / "mgrh_head.pt",
        mgrh_path / "pair_encoder.pt",
    ]:
        if not required.exists():
            errors.append(f"  Missing: {required}")
    if errors:
        logger.error("Missing required files:\n" + "\n".join(errors))
        sys.exit(1)

    logger.info("Input validation OK")
    logger.info(f"  Base checkpoint : {base_path}")
    logger.info(f"  MGRH checkpoint : {mgrh_path}")
    logger.info(f"  Output          : {output_path}")

    if args.dry_run:
        logger.info("Dry run — exiting before writing.")
        return

    # ------------------------------------------------------------------
    # Load YAML config for MGRH head architecture
    # ------------------------------------------------------------------
    mgrh_head_config: dict = {}
    if config_path.exists():
        cfg = load_yaml_config(config_path)
        mgrh_head_config = cfg.get("mgrh_training", {}).get("head", {})
        logger.info(f"  MGRH head config from {config_path}")
    else:
        logger.warning(f"  Config not found at {config_path}, using defaults")

    pair_enc_config = mgrh_head_config.get("pair_encoder", {})

    # ------------------------------------------------------------------
    # Imports (after sys.path is set)
    # ------------------------------------------------------------------
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoTokenizer

    from src.modeling_studio.models.heads import MultiGranularityRelevanceHead
    from src.modeling_studio.models.pair_encoder import CrossAttentionPairEncoder
    from src.modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    # ------------------------------------------------------------------
    # 1. Load base model (encoder + all frozen heads)
    # ------------------------------------------------------------------
    logger.info("Loading base model...")
    model_config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)

    # Load capabilities — add "relevance" if not present
    with open(base_path / "capabilities.json", encoding="utf-8") as f:
        cap_data = json.load(f)

    base_capabilities = cap_data.get("capabilities", [])

    # Import Capability enum to build the model
    from src.modeling_studio.models.modernbert_multitask import Capability

    capabilities = []
    for cap_str in base_capabilities:
        try:
            capabilities.append(Capability(cap_str))
        except ValueError:
            logger.warning(f"  Unknown capability string: {cap_str!r}, skipping")

    model = ModernBertMultiTaskModel(
        config=model_config,
        capabilities=capabilities,
        freeze_encoder=False,
    )

    # Restore checkpoint head architecture (GlobalPointer etc.)
    try:
        from scripts.training.train_embedding_heads_bakeoff import (
            restore_checkpoint_head_architecture,
            create_embedding_head,
            load_checkpoint_capabilities,
        )
        restore_checkpoint_head_architecture(model, base_path)
    except ImportError:
        logger.warning("  Could not import restore_checkpoint_head_architecture — heads may not match exactly")

    # Recreate embedding head from metadata so state_dict keys align
    emb_meta_path = base_path / "embedding_metadata.json"
    if emb_meta_path.exists():
        with open(emb_meta_path, encoding="utf-8") as f:
            emb_meta = json.load(f)
        bakeoff_info = emb_meta.get("bakeoff", {})
        emb_head_type = bakeoff_info.get("head_type", "agreement_gated_v2")
        emb_head_params = bakeoff_info.get("head_params", {})
        from scripts.training.train_embedding_heads_bakeoff import create_embedding_head
        emb_head = create_embedding_head(
            head_type=emb_head_type,
            hidden_size=model_config.hidden_size,
            **emb_head_params,
        )
        model.heads["embedding"] = emb_head
        logger.info(f"  Embedding head: {emb_head_type}")

    # Initialize encoder weights (required before load_state_dict)
    model._init_encoder()

    # Load full weights from base checkpoint
    state_dict = load_file(str(base_path / "model.safetensors"))
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state_dict.items() if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"  Loaded encoder: {len(encoder_state)} tensors")

    loaded_heads = []
    for head_name in list(model.heads.keys()):
        prefix = f"heads.{head_name}."
        head_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if head_state:
            try:
                model.heads[head_name].load_state_dict(head_state, strict=True)
                loaded_heads.append(head_name)
            except Exception as e:
                logger.warning(f"  Could not load {head_name} head: {e}")
    logger.info(f"  Loaded heads: {', '.join(loaded_heads)}")

    # ------------------------------------------------------------------
    # 2. Instantiate MGRH head with same architecture used during training
    # ------------------------------------------------------------------
    logger.info("Instantiating MGRH head...")
    pair_encoder = CrossAttentionPairEncoder(
        hidden_size=model_config.hidden_size,
        num_heads=pair_enc_config.get("num_heads", 8),
        num_layers=pair_enc_config.get("num_layers", 2),
        dropout=pair_enc_config.get("dropout", 0.1),
        use_bidirectional=True,
        pooling_strategy="attention",
    )
    mgrh_head = MultiGranularityRelevanceHead(
        hidden_size=model_config.hidden_size,
        dropout=mgrh_head_config.get("dropout", 0.1),
        pair_encoder=pair_encoder,
    )

    # ------------------------------------------------------------------
    # 3. Load MGRH weights — mgrh_head.pt was saved from EMA model
    # ------------------------------------------------------------------
    logger.info("Loading MGRH weights from checkpoint (EMA)...")
    head_state = torch.load(mgrh_path / "mgrh_head.pt", map_location="cpu", weights_only=True)
    mgrh_head.load_state_dict(head_state, strict=True)
    logger.info(f"  MGRH head: {sum(p.numel() for p in mgrh_head.parameters()):,} params")

    # Verify pair_encoder.pt is consistent with mgrh_head.pt
    # (pair_encoder is a submodule of mgrh_head — both should already be in sync)
    pe_state = torch.load(mgrh_path / "pair_encoder.pt", map_location="cpu", weights_only=True)
    mgrh_state_pe = {
        k.replace("pair_encoder.", "", 1): v
        for k, v in head_state.items() if k.startswith("pair_encoder.")
    }
    mismatch = [k for k in pe_state if not torch.equal(pe_state[k], mgrh_state_pe.get(k, torch.tensor([])))]
    if mismatch:
        logger.warning(
            f"  pair_encoder.pt has {len(mismatch)} tensors that differ from mgrh_head.pt — "
            f"using mgrh_head.pt (EMA). This is expected if pair_encoder.pt was not EMA-saved."
        )
    else:
        logger.info("  pair_encoder.pt is consistent with mgrh_head.pt")

    # ------------------------------------------------------------------
    # 4. Attach MGRH head to model under "relevance" key
    # ------------------------------------------------------------------
    model.heads["relevance"] = mgrh_head
    model.pair_encoder = pair_encoder
    logger.info("  Attached MGRH head as model.heads['relevance']")

    # ------------------------------------------------------------------
    # 5. Build merged state dict and save
    # ------------------------------------------------------------------
    logger.info("Building merged state dict...")
    merged: dict[str, torch.Tensor] = {}

    for name, param in model.encoder.state_dict().items():
        merged[f"encoder.{name}"] = param.cpu()

    for head_name, head in model.heads.items():
        for name, param in head.state_dict().items():
            merged[f"heads.{head_name}.{name}"] = param.cpu()

    # Also save pair_encoder at top level for load_checkpoint() compat
    if hasattr(model, "pair_encoder") and model.pair_encoder is not None:
        for name, param in model.pair_encoder.state_dict().items():
            merged[f"pair_encoder.{name}"] = param.cpu()

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model.safetensors ({len(merged)} tensors)...")
    save_file(merged, output_path / "model.safetensors")

    # ------------------------------------------------------------------
    # 6. Copy config + tokenizer from base checkpoint
    # ------------------------------------------------------------------
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "globalpointer_metadata.json",
        "embedding_metadata.json",
    ]:
        src = base_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            logger.info(f"  Copied {fname}")

    # ------------------------------------------------------------------
    # 7. Write updated capabilities.json (add "relevance")
    # ------------------------------------------------------------------
    updated_caps = list(base_capabilities)
    if "relevance" not in updated_caps:
        updated_caps.append("relevance")

    merged_cap_data = {
        "capabilities": updated_caps,
        "decoder_type": cap_data.get("decoder_type"),
        "epic_5_0": {
            **cap_data.get("epic_5_0", {}),
            "use_pair_encoder": True,
            "pair_encoder_num_layers": pair_enc_config.get("num_layers", 2),
        },
    }
    with open(output_path / "capabilities.json", "w", encoding="utf-8") as f:
        json.dump(merged_cap_data, f, indent=2)
    logger.info("  Wrote capabilities.json (added 'relevance')")

    # ------------------------------------------------------------------
    # 8. Write mgrh_metadata.json
    # ------------------------------------------------------------------
    # Load calibration temperature — try calibration.pt first, then mgrh_metadata.json
    temperature = None
    maxsim_pop_mean = None
    maxsim_pop_std = None
    calibration_file = mgrh_path.parent / "final" / "calibration.pt"
    if calibration_file.exists():
        cal = torch.load(calibration_file, map_location="cpu", weights_only=True)
        temperature = float(cal.get("temperature", 1.0))
    # Also read mgrh_metadata.json from stage output for MaxSim stats + temperature fallback
    stage_meta_file = mgrh_path.parent / "mgrh_metadata.json"
    if stage_meta_file.exists():
        with open(stage_meta_file, encoding="utf-8") as f:
            stage_meta = json.load(f)
        cal_info = stage_meta.get("calibration", {})
        if temperature is None and cal_info.get("temperature") is not None:
            temperature = float(cal_info["temperature"])
        maxsim_pop_mean = cal_info.get("maxsim_population_mean")
        maxsim_pop_std = cal_info.get("maxsim_population_std")

    # Load best metrics
    best_metrics = {}
    metrics_file = mgrh_path / "mgrh_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            best_metrics = json.load(f)

    mgrh_meta = {
        "timestamp": datetime.now().isoformat(),
        "source": {
            "base_checkpoint": str(base_path),
            "mgrh_checkpoint": str(mgrh_path),
            "weights": "EMA (mgrh_head.pt saved from ema_model.module)",
        },
        "architecture": {
            "head_type": "multi_granularity_relevance",
            "hidden_size": model_config.hidden_size,
            "dropout": mgrh_head_config.get("dropout", 0.1),
            "pair_encoder": {
                "num_heads": pair_enc_config.get("num_heads", 8),
                "num_layers": pair_enc_config.get("num_layers", 2),
                "dropout": pair_enc_config.get("dropout", 0.1),
                "use_bidirectional": True,
                "pooling_strategy": "attention",
            },
            "use_asymmetric_embeddings": mgrh_head_config.get("use_asymmetric_embeddings", True),
            "use_maxsim": mgrh_head_config.get("use_maxsim", True),
            "use_domain_saliency": mgrh_head_config.get("use_domain_saliency", True),
        },
        "calibration": {
            "temperature": temperature,
            "maxsim_population_mean": maxsim_pop_mean,
            "maxsim_population_std": maxsim_pop_std,
        },
        "best_metrics": best_metrics,
        "head_key": "relevance",
        "total_params": sum(p.numel() for p in mgrh_head.parameters()),
        "pair_encoder_params": sum(p.numel() for p in pair_encoder.parameters()),
    }
    with open(output_path / "mgrh_metadata.json", "w", encoding="utf-8") as f:
        json.dump(mgrh_meta, f, indent=2)
    logger.info("  Wrote mgrh_metadata.json")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("")
    logger.info("=" * 60)
    logger.info("Merged checkpoint saved successfully")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Heads: {', '.join(model.heads.keys())}")
    if temperature is not None:
        logger.info(f"  Calibration temperature: {temperature:.4f}")
    if best_metrics:
        logger.info(f"  Stage C best metrics:")
        for k, v in best_metrics.items():
            logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
