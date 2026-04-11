import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug the structured preference data/model pipeline before training."
    )
    parser.add_argument("--data_dir", default="data/synthetic_v1_1")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--val_file", default="val.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--preview_count", type=int, default=2)
    return parser.parse_args()


def check_modules(module_names):
    status = {}
    for name in module_names:
        try:
            importlib.import_module(name)
            status[name] = "ok"
        except Exception as exc:  # pragma: no cover
            status[name] = f"missing: {type(exc).__name__}: {exc}"
    return status


def print_status(title, mapping):
    print(f"\n{title}")
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    print(f"python executable: {sys.executable}")
    print(f"python version: {sys.version.split()[0]}")

    train_path = Path(args.data_dir) / args.train_file
    val_path = Path(args.data_dir) / args.val_file
    print(f"train path: {train_path}")
    print(f"val path: {val_path}")
    print(f"train exists: {train_path.exists()}")
    print(f"val exists: {val_path.exists()}")

    runtime_modules = ["jax", "flax", "optax", "transformers", "ml_collections"]
    print_status("Runtime dependency check", check_modules(runtime_modules))

    from JaxPref.data.structured_pref_dataset import StructuredPrefDataset, load_structured_pref_batch

    train_dataset = StructuredPrefDataset(str(train_path), max_seq_len=args.max_seq_len)
    val_dataset = StructuredPrefDataset(str(val_path), max_seq_len=args.max_seq_len)
    print("\nLoader check")
    print(f"  train samples: {len(train_dataset)}")
    print(f"  val samples: {len(val_dataset)}")
    print(f"  observation_dim: {train_dataset.observation_dim}")
    print(f"  action_dim: {train_dataset.action_dim}")

    preview = train_dataset.get_batch(range(min(args.preview_count, len(train_dataset))))
    print("  preview observations shape:", preview["observations"].shape)
    print("  preview actions shape:", preview["actions"].shape)
    print("  preview labels shape:", preview["labels"].shape)
    print("  preview reason labels:", preview["reason_label"].tolist())
    print("  preview reaction labels:", preview["reaction_label"].tolist())
    print("  preview dominant labels:", preview["dominant_dimension_label"].tolist())
    print("  preview secondary labels:", preview["secondary_dimension_label"].tolist())
    print("  preview decomp better labels shape:", preview["decomp_better_labels"].shape)
    print("  preview decomp score targets shape:", preview["decomp_score_targets"].shape)
    print("  preview sample ids:", list(preview["sample_id"]))

    prefmmt_batch = load_structured_pref_batch(str(val_path), max_seq_len=args.max_seq_len)
    print("\nPrefMMT-compatible batch check")
    print("  observations:", prefmmt_batch["observations"].shape)
    print("  actions:", prefmmt_batch["actions"].shape)
    print("  observations_2:", prefmmt_batch["observations_2"].shape)
    print("  actions_2:", prefmmt_batch["actions_2"].shape)
    print("  timestep_1:", prefmmt_batch["timestep_1"].shape)
    print("  attn_mask:", prefmmt_batch["attn_mask"].shape)
    print("  dimension_scores:", prefmmt_batch["dimension_scores"].shape)
    print("  decomp_better_labels:", prefmmt_batch["decomp_better_labels"].shape)
    print("  decomp_score_targets:", prefmmt_batch["decomp_score_targets"].shape)

    model_modules = ["JaxPref.PrefMMT", "flaxmodels.flaxmodels.gpt2.trajectory_gpt2"]
    print_status("Model import check", check_modules(model_modules))

    try:
        from JaxPref.PrefMMT import PrefMMT

        config = PrefMMT.get_default_config()
        print("\nModel config check")
        print(f"  lambda_reason: {config.lambda_reason}")
        print(f"  lambda_reaction: {config.lambda_reaction}")
        print(f"  lambda_dominant: {config.lambda_dominant}")
        print(f"  lambda_secondary: {config.lambda_secondary}")
        print(f"  lambda_decomp: {config.lambda_decomp}")
        print(f"  lambda_decomp_scores: {config.lambda_decomp_scores}")
        print(f"  lambda_dom_sec_consistency: {config.lambda_dom_sec_consistency}")
        print(f"  lambda_decomp_consistency: {config.lambda_decomp_consistency}")
        print(f"  lambda_priority_veto: {config.lambda_priority_veto}")
        print(f"  num_reason_classes: {config.num_reason_classes}")
        print(f"  num_reaction_classes: {config.num_reaction_classes}")
        print(f"  num_dominant_classes: {config.num_dominant_classes}")
        print(f"  num_secondary_classes: {config.num_secondary_classes}")
        print(f"  num_decomp_classes: {config.num_decomp_classes}")
    except Exception as exc:  # pragma: no cover
        print("\nModel config check")
        print(f"  skipped: {type(exc).__name__}: {exc}")

    print("\nStructured preference pipeline debug finished.")


if __name__ == "__main__":
    main()
