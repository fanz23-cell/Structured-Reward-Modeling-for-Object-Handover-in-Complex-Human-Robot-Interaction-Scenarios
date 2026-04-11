import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check V2 plus synthetic data quality gates.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--preview-count", type=int, default=3)
    return parser.parse_args()


def load_records(path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expand_input(path):
    if path.is_dir():
        return [p for p in [path / "train.jsonl", path / "val.jsonl"] if p.exists()]
    return [path]


def better_label(a, b, tol=0.03):
    if a > b + tol:
        return "A"
    if b > a + tol:
        return "B"
    return "tie"


def structured_score(comfort, safety, tau_safe=0.72, lambda_veto=4.0):
    penalty = max(tau_safe - safety, 0.0)
    return comfort - lambda_veto * penalty * penalty


def main_subreason(controls):
    items = [(k, float(v)) for k, v in controls.items() if k in {"reachability", "height_alignment", "front_sector_alignment", "posture_cost", "distance_intrusion", "speed_risk", "timing_risk"}]
    items.sort(key=lambda item: item[1], reverse=True)
    return items[0][0] if items else "unclear"


def reaction_from_reason(reason_label, safety_subreason, comfort_subreason):
    if reason_label == "safety":
        return "interruption" if safety_subreason == "timing_risk" else "avoidance"
    if reason_label == "comfort":
        return "unnatural_posture" if comfort_subreason == "height_alignment" else "overreach_or_reposition"
    if reason_label == "mixed":
        return "hesitation"
    return "unclear"


def check_record(record, errors, stats, previews):
    candidate_lookup = {c["candidate_id"]: c for c in record["candidate_set"]}
    stats["scene_type"][record["context"]["scene_type"]] += 1
    for candidate in record["candidate_set"]:
        stats["candidate_source"][candidate["candidate_source"]] += 1

    for pair in record["training_pairs"]:
        a = candidate_lookup[pair["candidate_a_id"]]
        b = candidate_lookup[pair["candidate_b_id"]]
        labels = pair["labels"]
        stats["overall_preference"][labels["overall_preference"]] += 1
        stats["reason_label"][labels["reason_label"]] += 1
        stats["reaction_label"][labels["reaction_label"]] += 1
        stats["comfort_better_label"][labels["comfort_better_label"]] += 1
        stats["safety_better_label"][labels["safety_better_label"]] += 1
        if labels["overall_preference"] in {"tie", "unclear"}:
            stats["tie_unclear"][labels["overall_preference"]] += 1

        comfort_a = a["synthetic_scores"]["comfort_score_target"]
        comfort_b = b["synthetic_scores"]["comfort_score_target"]
        safety_a = a["synthetic_scores"]["safety_score_target"]
        safety_b = b["synthetic_scores"]["safety_score_target"]
        comfort_better = better_label(comfort_a, comfort_b)
        safety_better = better_label(safety_a, safety_b)
        overall = better_label(structured_score(comfort_a, safety_a), structured_score(comfort_b, safety_b), tol=0.025)
        if comfort_better != labels["comfort_better_label"]:
            errors.append(f"{record['context_id']}::{pair['pair_id']} comfort_better mismatch")
        if safety_better != labels["safety_better_label"]:
            errors.append(f"{record['context_id']}::{pair['pair_id']} safety_better mismatch")
        if overall != labels["overall_preference"]:
            errors.append(f"{record['context_id']}::{pair['pair_id']} overall_preference violates safety->comfort hierarchy")

        loser = b if labels["overall_preference"] == "A" else a
        winner = a if loser is b else b
        safety_gap = winner["synthetic_scores"]["safety_score_target"] - loser["synthetic_scores"]["safety_score_target"]
        comfort_gap = winner["synthetic_scores"]["comfort_score_target"] - loser["synthetic_scores"]["comfort_score_target"]
        expected_reason = "unclear" if abs(safety_gap) < 0.04 and abs(comfort_gap) < 0.04 else ("safety" if loser["synthetic_scores"]["safety_score_target"] < 0.62 and safety_gap >= comfort_gap - 0.02 else ("comfort" if loser["synthetic_scores"]["comfort_score_target"] < 0.62 and comfort_gap > safety_gap + 0.02 else "mixed"))
        if labels["reason_label"] != expected_reason:
            errors.append(f"{record['context_id']}::{pair['pair_id']} reason_label not aligned with loser main defect")

        expected_reaction = reaction_from_reason(
            labels["reason_label"],
            loser["synthetic_scores"]["safety_subreason_label"],
            loser["synthetic_scores"]["comfort_subreason_label"],
        )
        if labels["reaction_label"] != expected_reaction:
            errors.append(f"{record['context_id']}::{pair['pair_id']} reaction_label not aligned with defects")

        if labels.get("safety_subreason_label") is not None:
            expected_safety_subreason = main_subreason(loser["synthetic_controls"]["safety_controls"])
            if expected_safety_subreason != labels["safety_subreason_label"]:
                errors.append(f"{record['context_id']}::{pair['pair_id']} safety_subreason mismatch")
        if labels.get("comfort_subreason_label") is not None:
            expected_comfort_subreason = main_subreason(loser["synthetic_controls"]["comfort_controls"])
            if expected_comfort_subreason != labels["comfort_subreason_label"]:
                errors.append(f"{record['context_id']}::{pair['pair_id']} comfort_subreason mismatch")

        previews.append(
            {
                "context_id": record["context_id"],
                "pair_id": pair["pair_id"],
                "scene_type": record["context"]["scene_type"],
                "candidate_sources": [a["candidate_source"], b["candidate_source"]],
                "comfort_scores": [comfort_a, comfort_b],
                "safety_scores": [safety_a, safety_b],
                "overall_preference": labels["overall_preference"],
                "reason_label": labels["reason_label"],
                "reaction_label": labels["reaction_label"],
                "loser_safety_subreason": loser["synthetic_scores"]["safety_subreason_label"],
                "loser_comfort_subreason": loser["synthetic_scores"]["comfort_subreason_label"],
            }
        )


def main():
    args = parse_args()
    files = expand_input(Path(args.data))
    errors = []
    stats = {
        "scene_type": Counter(),
        "candidate_source": Counter(),
        "overall_preference": Counter(),
        "reason_label": Counter(),
        "reaction_label": Counter(),
        "comfort_better_label": Counter(),
        "safety_better_label": Counter(),
        "tie_unclear": Counter(),
    }
    previews = []
    for path in files:
        records = load_records(path)
        for record in records:
            check_record(record, errors, stats, previews)

    if errors:
        print("QUALITY GATE FAIL")
        for err in errors[:50]:
            print(" -", err)
        raise SystemExit(1)

    required_reason_labels = {"safety", "comfort", "mixed", "unclear"}
    observed_reason_labels = set(stats["reason_label"].keys())
    if not required_reason_labels.issubset(observed_reason_labels):
        missing = sorted(required_reason_labels - observed_reason_labels)
        print("QUALITY GATE FAIL")
        print(f" - missing reason label coverage: {missing}")
        raise SystemExit(1)

    required_reactions = {"avoidance", "hesitation", "overreach_or_reposition", "unnatural_posture", "interruption"}
    observed_reactions = set(stats["reaction_label"].keys())
    if not required_reactions.issubset(observed_reactions):
        missing = sorted(required_reactions - observed_reactions)
        print("QUALITY GATE FAIL")
        print(f" - missing reaction label coverage: {missing}")
        raise SystemExit(1)

    print("QUALITY GATE PASS")
    print("\nDistributions")
    for key, counter in stats.items():
        print(f"  {key}: {dict(sorted(counter.items()))}")
    print(f"\nPreviewing {min(args.preview_count, len(previews))} pair samples")
    for item in previews[: args.preview_count]:
        print(" ", item)


if __name__ == "__main__":
    main()
