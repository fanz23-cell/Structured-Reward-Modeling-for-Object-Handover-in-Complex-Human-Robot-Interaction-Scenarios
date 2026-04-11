import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))


def parse_args():
    parser = argparse.ArgumentParser(description="Validate weakly-real V2 template compatibility.")
    parser.add_argument("--data", default="data/weakly_real_v2_template/example_weakly_real_context.json")
    parser.add_argument("--max_seq_len", type=int, default=8)
    return parser.parse_args()


def main():
    from check_real_pref_v2_cs_rethinking import _resolve_to_jsonl
    from JaxPref.data.structured_pref_dataset_v2_cs_rethinking import StructuredPrefDatasetV2CSRethinking

    args = parse_args()
    jsonl_path = _resolve_to_jsonl(Path(args.data))
    dataset = StructuredPrefDatasetV2CSRethinking(str(jsonl_path), max_seq_len=args.max_seq_len)
    batch = dataset.get_batch(range(1))
    print("Weakly-real V2 template check")
    print(f"  source: {args.data}")
    print(f"  pairs: {len(dataset)}")
    print(f"  context_feature_dim: {dataset.context_dim}")
    print(f"  geometry_raw_feature_dim: {dataset.geometry_raw_dim}")
    print(f"  comfort_score_masks: {batch['comfort_score_masks'][0].tolist()}")
    print(f"  safety_score_masks: {batch['safety_score_masks'][0].tolist()}")
    print(f"  safety_subreason_mask: {float(batch['safety_subreason_mask'][0])}")
    print(f"  comfort_subreason_mask: {float(batch['comfort_subreason_mask'][0])}")


if __name__ == "__main__":
    main()
