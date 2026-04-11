import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))


OPTIONAL_LABEL_KEYS = [
    "comfort_score_target",
    "safety_score_target",
    "safety_subreason_label",
    "comfort_subreason_label",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check V2 readiness for real-data cases with missing optional targets."
    )
    parser.add_argument("--source_dir", default="data/synthetic_v2_cs_rethinking")
    parser.add_argument(
        "--output_root",
        default="data/synthetic_v2_cs_rethinking_missing_optional_checks",
    )
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--geometry_mode", choices=["fixed", "trainable"], default="trainable")
    parser.add_argument(
        "--geometry_parameterization_mode",
        choices=["global", "contextual"],
        default="contextual",
    )
    parser.add_argument("--run_train_smoke", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dump_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def rewrite_records(records, scenario):
    rewritten = []
    for record in records:
        updated_record = dict(record)
        updated_pairs = []
        for pair in record["training_pairs"]:
            updated_pair = dict(pair)
            labels = dict(pair["labels"])
            if scenario in {"scores_missing", "all_optional_missing"}:
                labels["comfort_score_target"] = None
                labels["safety_score_target"] = None
            if scenario in {"subreasons_missing", "all_optional_missing"}:
                labels["safety_subreason_label"] = None
                labels["comfort_subreason_label"] = None
            updated_pair["labels"] = labels
            updated_pairs.append(updated_pair)
        updated_record["training_pairs"] = updated_pairs
        rewritten.append(updated_record)
    return rewritten


def summarize_dataset(dataset):
    comfort_mask_sum = float(np.sum([sample["comfort_score_masks"] for sample in dataset.samples]))
    safety_mask_sum = float(np.sum([sample["safety_score_masks"] for sample in dataset.samples]))
    safety_subreason_mask_sum = float(np.sum([sample["safety_subreason_mask"] for sample in dataset.samples]))
    comfort_subreason_mask_sum = float(np.sum([sample["comfort_subreason_mask"] for sample in dataset.samples]))
    return {
        "pairs": len(dataset),
        "comfort_score_mask_sum": comfort_mask_sum,
        "safety_score_mask_sum": safety_mask_sum,
        "safety_subreason_mask_sum": safety_subreason_mask_sum,
        "comfort_subreason_mask_sum": comfort_subreason_mask_sum,
    }


def run_check_script(data_dir: Path):
    cmd = [
        sys.executable,
        "scripts/check_structured_pref_data_v2_cs_rethinking.py",
        "--data",
        str(data_dir),
        "--preview-count",
        "1",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def run_train_smoke(data_dir: Path, geometry_mode: str, geometry_parameterization_mode: str):
    cmd = [
        sys.executable,
        "scripts/train_structured_pref_v2_cs_rethinking.py",
        "--data_dir",
        str(data_dir),
        "--epochs",
        "1",
        "--batch_size",
        "4",
        "--max_seq_len",
        "8",
        "--device",
        "cpu",
        "--debug_batches",
        "1",
        "--geometry_mode",
        geometry_mode,
        "--geometry_parameterization_mode",
        geometry_parameterization_mode,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    args = parse_args()

    from JaxPref.data.structured_pref_dataset_v2_cs_rethinking import StructuredPrefDatasetV2CSRethinking

    source_dir = Path(args.source_dir)
    train_records = load_jsonl(source_dir / "train.jsonl")
    val_records = load_jsonl(source_dir / "val.jsonl")

    scenarios = [
        "scores_missing",
        "subreasons_missing",
        "all_optional_missing",
    ]

    print("Missing-target readiness check")
    print(f"  source_dir: {source_dir}")
    print(f"  output_root: {args.output_root}")
    print(f"  geometry_mode: {args.geometry_mode}")
    print(f"  geometry_parameterization_mode: {args.geometry_parameterization_mode}")

    for scenario in scenarios:
        scenario_dir = Path(args.output_root) / scenario
        dump_jsonl(scenario_dir / "train.jsonl", rewrite_records(train_records, scenario))
        dump_jsonl(scenario_dir / "val.jsonl", rewrite_records(val_records, scenario))

        train_dataset = StructuredPrefDatasetV2CSRethinking(
            str(scenario_dir / "train.jsonl"),
            max_seq_len=args.max_seq_len,
        )
        val_dataset = StructuredPrefDatasetV2CSRethinking(
            str(scenario_dir / "val.jsonl"),
            max_seq_len=args.max_seq_len,
        )
        check_code, _, check_stderr = run_check_script(scenario_dir)

        print(f"\nscenario: {scenario}")
        print(f"  train_summary: {summarize_dataset(train_dataset)}")
        print(f"  val_summary: {summarize_dataset(val_dataset)}")
        print(f"  schema_check_ok: {check_code == 0}")
        if check_code != 0:
            print(f"  schema_check_stderr: {check_stderr.strip()}")

        if args.run_train_smoke:
            train_code, train_stdout, train_stderr = run_train_smoke(
                scenario_dir,
                args.geometry_mode,
                args.geometry_parameterization_mode,
            )
            print(f"  train_smoke_ok: {train_code == 0}")
            if train_code == 0:
                lines = [line for line in train_stdout.splitlines() if line.startswith("train ") or line.startswith("val ")]
                for line in lines[-2:]:
                    print(f"  {line}")
            else:
                print(f"  train_smoke_stderr: {train_stderr.strip()}")


if __name__ == "__main__":
    main()
