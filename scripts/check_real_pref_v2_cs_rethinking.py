import argparse
import json
import tempfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a real-data V2 comfort/safety structured preference template."
    )
    parser.add_argument(
        "--data",
        default="data/real_pref_v2_template/example_real_context.json",
        help="Path to a single .json template record or a .jsonl file/directory compatible with the V2 loader.",
    )
    parser.add_argument("--max_seq_len", type=int, default=8)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _resolve_to_jsonl(data_path: Path) -> Path:
    if data_path.is_dir():
        return data_path / "train.jsonl"
    if data_path.suffix == ".jsonl":
        return data_path
    if data_path.suffix != ".json":
        raise ValueError(f"Unsupported input format: {data_path}")

    record = _load_json(data_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="real_pref_v2_check_"))
    tmp_path = tmp_dir / "example.jsonl"
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    return tmp_path


def main():
    args = parse_args()

    from JaxPref.data.structured_pref_dataset_v2_cs_rethinking import StructuredPrefDatasetV2CSRethinking

    data_path = Path(args.data)
    jsonl_path = _resolve_to_jsonl(data_path)
    dataset = StructuredPrefDatasetV2CSRethinking(str(jsonl_path), max_seq_len=args.max_seq_len)
    batch = dataset.get_batch(range(min(len(dataset), 1)))

    print("Real-data V2 template check")
    print(f"  source: {data_path}")
    print(f"  resolved_jsonl: {jsonl_path}")
    print(f"  pairs: {len(dataset)}")
    print(f"  observation_dim: {dataset.observation_dim}")
    print(f"  action_dim: {dataset.action_dim}")
    print(f"  context_feature_dim: {dataset.context_dim}")
    print(f"  geometry_feature_dim: {dataset.geometry_dim}")
    print(f"  geometry_raw_feature_dim: {dataset.geometry_raw_dim}")
    print(f"  sample_id: {batch['sample_id'][0]}")
    print(f"  comfort_score_masks: {batch['comfort_score_masks'][0].tolist()}")
    print(f"  safety_score_masks: {batch['safety_score_masks'][0].tolist()}")
    print(f"  safety_subreason_mask: {float(batch['safety_subreason_mask'][0])}")
    print(f"  comfort_subreason_mask: {float(batch['comfort_subreason_mask'][0])}")


if __name__ == "__main__":
    main()
