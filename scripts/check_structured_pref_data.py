import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_TOP_LEVEL = [
    "sample_id",
    "pair_metadata",
    "segment_a",
    "segment_b",
    "labels",
    "reward_decomposition",
    "annotator_id",
    "notes",
]
REQUIRED_SEGMENT = [
    "segment_id",
    "scene_type",
    "segment_start_type",
    "segment_end_type",
    "handover_outcome",
    "candidate_source",
    "candidate_generator_detail",
    "planner_family",
    "hybrid_role",
    "source_description",
    "context",
    "sequences",
]
REQUIRED_CONTEXT = [
    "map_id",
    "handover_point_3d",
    "handover_point_relative_to_human",
    "handover_point_relative_to_robot",
    "handover_point_map_zone",
    "human_initial_relative_position",
    "robot_initial_relative_position",
    "scene_notes",
]
REQUIRED_SEQUENCE = [
    "time_seq",
    "robot_state_seq",
    "robot_action_seq",
    "base_pose_seq",
    "ee_pose_seq",
    "velocity_seq",
    "gripper_state_seq",
    "object_pose_seq",
    "human_pose_seq",
    "human_hand_pose_seq",
    "relative_position_seq",
    "relative_distance_seq",
    "relative_orientation_seq",
]
REQUIRED_LABELS = [
    "overall_preference",
    "reason_label",
    "reaction_label",
    "winner_outcome_alignment",
    "dominant_dimension",
    "secondary_dimension",
]
REQUIRED_DIMENSION_FIELDS = [
    "score_a",
    "score_b",
    "better_segment",
    "severity_if_bad",
    "confidence",
    "notes",
]
PAIR_METADATA_ENUMS = {
    "pair_generation_protocol": {
        "same_source_variation",
        "cross_source_comparison",
        "hybrid_ablation",
        "manual_baseline_pair",
    },
    "comparison_goal": {
        "classical_vs_learned",
        "classical_vs_hybrid",
        "learned_vs_hybrid",
        "safe_vs_efficient",
        "comfortable_vs_efficient",
        "social_vs_direct",
        "other",
    },
    "pair_scene_control": {
        "same_scene_same_handover_goal",
        "same_scene_different_handover_style",
        "different_scene_exploratory",
    },
}
SEGMENT_ENUMS = {
    "scene_type": {
        "open_space",
        "narrow_corridor",
        "doorway",
        "corner",
        "intersection",
        "enclosed_space",
        "table_separated",
    },
    "segment_start_type": {
        "human_notice",
        "human_first_reaction",
        "robot_enter_interaction_zone",
    },
    "segment_end_type": {
        "handover_success",
        "handover_failure",
        "handover_abort",
    },
    "handover_outcome": {"success", "failure", "partial", "unclear"},
    "candidate_source": {"classical", "learned", "hybrid", "manual_baseline"},
    "candidate_generator_detail": {
        "rule_based_safe",
        "rule_based_efficient",
        "rule_based_social_nominal",
        "learned_social_style",
        "learned_direct_style",
        "hybrid_switch",
        "hybrid_blend",
        "manual_good_case",
        "manual_bad_case",
        "manual_variant",
    },
    "planner_family": {"geometric", "learning_based", "hybrid", "manual"},
    "hybrid_role": {
        "none",
        "geometric_backbone_learning_refinement",
        "learning_backbone_geometric_guard",
        "switching_controller",
    },
}
LABEL_ENUMS = {
    "overall_preference": {"A", "B", "tie"},
    "reason_label": {"safety", "comfort", "social", "efficiency", "unclear"},
    "reaction_label": {
        "none",
        "hesitation",
        "avoidance",
        "interruption",
        "overreach_or_reposition",
        "unnatural_posture",
        "unclear",
    },
    "winner_outcome_alignment": {"aligned", "conflicted", "unclear"},
    "dominant_dimension": {"safety", "comfort", "social", "efficiency", "unclear"},
    "secondary_dimension": {"none", "safety", "comfort", "social", "efficiency", "unclear"},
}
DIMENSION_NAMES = ["safety", "comfort", "social", "efficiency"]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate structured preference synthetic data.")
    parser.add_argument(
        "--data",
        required=True,
        help="JSON/JSONL file or a directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument("--preview-count", type=int, default=3)
    return parser.parse_args()


def load_records(path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, list) else [payload]
    raise ValueError(f"Unsupported file type: {path}")


def ensure_fields(obj, required_fields, prefix, errors):
    for field in required_fields:
        if field not in obj:
            errors.append(f"{prefix}: missing field `{field}`")


def validate_enum(obj, field, allowed, prefix, errors):
    value = obj.get(field)
    if value not in allowed:
        errors.append(f"{prefix}: invalid `{field}`={value!r}")


def validate_sequences(sample_id, segment_key, sequences, errors):
    ensure_fields(sequences, REQUIRED_SEQUENCE, f"{sample_id}.{segment_key}.sequences", errors)
    time_seq = sequences.get("time_seq", [])
    target_len = len(time_seq)
    if target_len == 0:
        errors.append(f"{sample_id}.{segment_key}.sequences: empty time_seq")
        return
    for key in REQUIRED_SEQUENCE[1:]:
        seq = sequences.get(key)
        if not isinstance(seq, list):
            errors.append(f"{sample_id}.{segment_key}.sequences.{key}: must be a list")
            continue
        if len(seq) != target_len:
            errors.append(
                f"{sample_id}.{segment_key}.sequences.{key}: length {len(seq)} != time_seq length {target_len}"
            )


def validate_reward_decomposition(sample_id, reward_decomposition, errors):
    if "dimensions" not in reward_decomposition:
        errors.append(f"{sample_id}.reward_decomposition: missing `dimensions`")
        return
    dims = reward_decomposition["dimensions"]
    for name in DIMENSION_NAMES:
        if name not in dims:
            errors.append(f"{sample_id}.reward_decomposition.dimensions: missing `{name}`")
            continue
        ensure_fields(
            dims[name],
            REQUIRED_DIMENSION_FIELDS,
            f"{sample_id}.reward_decomposition.dimensions.{name}",
            errors,
        )
        better = dims[name].get("better_segment")
        if better not in {"A", "B", "tie", "unclear"}:
            errors.append(
                f"{sample_id}.reward_decomposition.dimensions.{name}.better_segment invalid: {better!r}"
            )
    if reward_decomposition.get("priority_order") != DIMENSION_NAMES:
        errors.append(
            f"{sample_id}.reward_decomposition.priority_order expected {DIMENSION_NAMES}, "
            f"got {reward_decomposition.get('priority_order')!r}"
        )
    if reward_decomposition.get("aggregation_rule") != "priority_weighted_with_safety_veto":
        errors.append(
            f"{sample_id}.reward_decomposition.aggregation_rule invalid: "
            f"{reward_decomposition.get('aggregation_rule')!r}"
        )


def validate_sample(sample, errors):
    sample_id = sample.get("sample_id", "<missing_id>")
    ensure_fields(sample, REQUIRED_TOP_LEVEL, sample_id, errors)

    pair_metadata = sample.get("pair_metadata", {})
    for field, allowed in PAIR_METADATA_ENUMS.items():
        validate_enum(pair_metadata, field, allowed, f"{sample_id}.pair_metadata", errors)

    for segment_key in ("segment_a", "segment_b"):
        segment = sample.get(segment_key, {})
        ensure_fields(segment, REQUIRED_SEGMENT, f"{sample_id}.{segment_key}", errors)
        for field, allowed in SEGMENT_ENUMS.items():
            validate_enum(segment, field, allowed, f"{sample_id}.{segment_key}", errors)
        context = segment.get("context", {})
        ensure_fields(context, REQUIRED_CONTEXT, f"{sample_id}.{segment_key}.context", errors)
        validate_sequences(sample_id, segment_key, segment.get("sequences", {}), errors)

    labels = sample.get("labels", {})
    ensure_fields(labels, REQUIRED_LABELS, f"{sample_id}.labels", errors)
    for field, allowed in LABEL_ENUMS.items():
        validate_enum(labels, field, allowed, f"{sample_id}.labels", errors)

    validate_reward_decomposition(sample_id, sample.get("reward_decomposition", {}), errors)


def summarize_records(records):
    stats = {
        "overall_preference": Counter(),
        "reason_label": Counter(),
        "reaction_label": Counter(),
        "candidate_source": Counter(),
        "scene_type": Counter(),
    }
    for sample in records:
        labels = sample["labels"]
        stats["overall_preference"][labels["overall_preference"]] += 1
        stats["reason_label"][labels["reason_label"]] += 1
        stats["reaction_label"][labels["reaction_label"]] += 1
        stats["scene_type"][sample["segment_a"]["scene_type"]] += 1
        stats["candidate_source"][sample["segment_a"]["candidate_source"]] += 1
        stats["candidate_source"][sample["segment_b"]["candidate_source"]] += 1
    return stats


def print_stats(title, stats):
    print(f"\n{title}")
    for key, counter in stats.items():
        print(f"  {key}:")
        for label, count in sorted(counter.items()):
            print(f"    {label}: {count}")


def print_preview(records, limit):
    print(f"\nPreviewing {min(limit, len(records))} samples")
    for sample in records[:limit]:
        print(f"- {sample['sample_id']}")
        print(
            f"  scene={sample['segment_a']['scene_type']} "
            f"A={sample['segment_a']['candidate_source']} "
            f"B={sample['segment_b']['candidate_source']}"
        )
        print(
            f"  preference={sample['labels']['overall_preference']} "
            f"reason={sample['labels']['reason_label']} "
            f"reaction={sample['labels']['reaction_label']}"
        )
        print(
            f"  dominant={sample['labels']['dominant_dimension']} "
            f"secondary={sample['labels']['secondary_dimension']}"
        )


def expand_input(path):
    if path.is_dir():
        candidates = [path / "train.jsonl", path / "val.jsonl"]
        return [item for item in candidates if item.exists()]
    return [path]


def main():
    args = parse_args()
    target = Path(args.data)
    files = expand_input(target)
    if not files:
        raise FileNotFoundError(f"No data files found under {target}")

    total_records = []
    all_errors = []
    for file_path in files:
        records = load_records(file_path)
        print(f"Loaded {len(records)} samples from {file_path}")
        for sample in records:
            validate_sample(sample, all_errors)
        stats = summarize_records(records)
        print_stats(f"Label summary for {file_path.name}", stats)
        print_preview(records, args.preview_count)
        total_records.extend(records)

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"\nValidation passed for {len(total_records)} samples across {len(files)} file(s).")


if __name__ == "__main__":
    main()
